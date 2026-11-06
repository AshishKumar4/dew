"""MiMo-V2.6's vision tower and audio path at a tiny scale, run by their references.

The fixtures tests/test_vision.py and tests/test_mimo_audio.py hold Dew to. The reference is
XiaomiMiMo/MiMo-V2.6-Pro-MOPD's `modeling_mimo_v2.py`, fetched at `MIMO_REVISION` (`mimo_module`),
except the log mel frames, which are torchaudio's at MiMo-Audio-Tokenizer's `mel_spectrogram` settings.
Each array file stores fp32 and float64 references, IEEE products.

- `mimo-vision-tiny`: the reference config, the ViT's and merger's tensors as the release names them,
  packed pixels with each row's grid, and `refs.npz`: the trunk (the merger's input) and the merger's
  output, the merger's biases zero as the release stores none.
- `mimo-audio-tiny/model.npz`: the tokenizer's encoder (AudioTokenizerEncoder, its codes by
  tokenize_audio_batch) and the audio encoder (MiMoAudioEncoder over the codes), with scattered
  weights: the inputs, the pooled features, the codes and the embeddings, once in the reference's one
  batch and once with its batches bounded at 30 frames, its own grouping run under that bound.
- `mimo-audio-tiny/mel.npz` (`--mel`): torchaudio stops at torch 2.11, so this part runs in an
  environment of torch 2.11 and torchaudio 2.11, the transform built in the dtype it computes in.
- `mimo-vision-tiny/recording.npz` (`--record-vision`): the fixture's tower and merger over drawn pixels
  (`vision_draw`) and drawn trunk features (`merger_draw`), their evidence and float64 truths' sketches
  recorded over seeds for check 1 (`recorded_references.record_case`, summarized).

    python tools/mimo_reference.py
    <torch 2.11 + torchaudio python> tools/mimo_reference.py --mel
    PYTHONPATH=src:tests python tools/mimo_reference.py --record-vision
    PYTHONPATH=src:tests python tools/mimo_reference.py --record-audio

`--record-audio` writes `mimo-audio-tiny/recording.npz`: the fixture's tokenizer encoder's pooled features
of drawn mels (`mels_draw`), its batches whole and bounded at 30 frames, and its audio encoder over drawn
codes (`codes_draw`), summarized over seeds for check 1.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "hf"
AUDIO_FIXTURE = FIXTURES / "mimo-audio-tiny"
MIMO_REPO = "XiaomiMiMo/MiMo-V2.6-Pro-MOPD"
MIMO_REVISION = "6a7f202e8dbd0e57b3fe1da03917d08d75abc4b8"

MIMO_VISION = dict(
    depth=4, hidden_size=32, intermediate_size=48, num_heads=4, num_key_value_heads=2, qk_channels=8,
    in_chans=3, patch_size=4, spatial_merge_size=2, temporal_patch_size=2, hidden_act="silu",
    rms_norm_eps=1e-6, fullatt_block_indexes=[0, 3], vit_window_attn_types=[-1, 0, 1, -1], use_sink=True,
    visual_token_window_size=3, out_hidden_size=16)
"""Two band layers, one in each token order, between two full ones, a band narrower than a frame's
rows, and sinks on the band layers."""

MIMO_GRIDS = ((1, 4, 6), (2, 4, 4))
"""A 4x6 image and a two-frame 4x4 clip, so the band crosses rows and the frames stay apart."""

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
convolved lengths, 10, 10, 3 and 7, put two odd ones beside the batch's longest; with the batches
bounded at 30 frames the segments batch as [20], [20, 5] and [13], the last alone and odd."""


def mimo_module():
    """MiMo's modeling module, fetched at MIMO_REVISION and imported as a package beside its config."""
    from reference_source import hub_modules

    _, modeling = hub_modules("mimo_v2", MIMO_REPO, MIMO_REVISION,
                              ("configuration_mimo_v2.py", "modeling_mimo_v2.py"))
    return modeling


def scatter(module: torch.nn.Module, seed: int) -> None:
    """Something in every tensor: norms near one, biases and sinks small, maps and codebooks random."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, tensor in [*module.named_parameters(), *module.named_buffers()]:
            if not tensor.is_floating_point() or "inv_freq" in name:
                continue
            if "norm" in name or "ln_" in name:
                tensor.copy_(1.0 + torch.randn(tensor.shape, generator=generator) * 0.05)
            elif "bias" in name or "sinks" in name:
                tensor.copy_(torch.randn(tensor.shape, generator=generator) * 0.2)
            else:
                tensor.copy_(torch.randn(tensor.shape, generator=generator) * 0.4)


def packed_rows(grids, width: int, seed: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Random packed pixels per row, `[patches, width]` each, and them padded into one batch."""
    generator = np.random.RandomState(seed)
    rows = [generator.randn(int(np.prod(grid)), width).astype(np.float32) for grid in grids]
    batch = np.zeros((len(rows), max(len(row) for row in rows), width), np.float32)
    for index, row in enumerate(rows):
        batch[index, :len(row)] = row
    return batch, rows


def padded(outputs: list[np.ndarray]) -> np.ndarray:
    """Per-row outputs `[n, ...]` as one zero-padded batch."""
    batch = np.zeros((len(outputs), max(len(x) for x in outputs), *outputs[0].shape[1:]), np.float64)
    for index, x in enumerate(outputs):
        batch[index, :len(x)] = x
    return batch


def write(directory: Path, config: dict, tower: dict, projector: dict, arrays: dict) -> None:
    from safetensors.torch import save_file

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    save_file({name: tensor.contiguous() for name, tensor in tower.items()}, directory / "model.safetensors")
    save_file({name: tensor.contiguous() for name, tensor in projector.items()}, directory / "projector.safetensors")
    np.savez(directory / "refs.npz", precision="ieee",
             **{name: np.asarray(value, np.float64 if name.endswith("_f64") else np.float32)
                for name, value in arrays.items() if not name.startswith("_")},
             **{name[1:]: value for name, value in arrays.items() if name.startswith("_")})
    print(directory, sorted(p.name for p in directory.iterdir()))


def write_mimo_vision_tiny() -> None:
    """MiMo ViT: the trunk sequence (the merger's input, caught by a hook) and the merger's output,
    the merger's biases zero as the release stores none."""
    from torch_precision import float64

    modeling = mimo_module()
    config = SimpleNamespace(**MIMO_VISION)
    torch.manual_seed(1234)
    tower = modeling.MiMoVisionTransformer(config).float().eval()
    scatter(tower, 1234)
    with torch.no_grad():
        for name, tensor in tower.merger.named_parameters():
            if name.endswith("bias"):
                tensor.zero_()
    width = 3 * MIMO_VISION["temporal_patch_size"] * MIMO_VISION["patch_size"] ** 2
    batch, rows = packed_rows(MIMO_GRIDS, width, 11)

    def run(model, dtype):
        trunks, merged = [], []
        caught = {}
        hook = model.merger.register_forward_hook(lambda module, inputs, output: caught.update(x=inputs[0]))
        with torch.no_grad():
            for row, grid in zip(rows, MIMO_GRIDS, strict=True):
                merged.append(model(torch.from_numpy(row).to(dtype), torch.tensor([grid])).numpy())
                trunks.append(caught["x"].numpy())
        hook.remove()
        return padded(trunks), padded(merged)

    trunk, merged = run(tower, torch.float32)
    with float64():
        wide = modeling.MiMoVisionTransformer(config).eval()
        wide.load_state_dict({name: tensor.double() for name, tensor in tower.state_dict().items()})
        trunk_f64, _ = run(wide, torch.float64)
        with torch.no_grad():
            merged_f64 = padded([wide.merger(torch.from_numpy(t[:len(r)]).double()).numpy()
                                 for t, r in zip(trunk.astype(np.float32), rows, strict=True)])
    state = tower.state_dict()
    write(FIXTURES / "mimo-vision-tiny",
          {"model_type": "mimo_v2", "vision_config": MIMO_VISION, "text_config": {"hidden_size": 16}},
          {name: tensor for name, tensor in state.items() if not name.startswith("merger.")},
          {name: tensor for name, tensor in state.items()
           if name.startswith("merger.") and not name.endswith("bias")},
          {"pixels": batch, "_grid": np.asarray(MIMO_GRIDS, np.int32), "tower": trunk, "tower_f64": trunk_f64,
           "projector": merged, "projector_f64": merged_f64})


VISION_WIDTH = 3 * MIMO_VISION["temporal_patch_size"] * MIMO_VISION["patch_size"] ** 2
"""Each packed patch's pixels: its channels, frames and rows of pixels."""


def vision_draw(seed: int) -> dict[str, np.ndarray]:
    """A seed's packed pixels of MIMO_GRIDS' two rows, padded into one batch (`packed_rows`), drawn apart
    from the fixture's own (seed 11)."""
    return {"pixels": packed_rows(MIMO_GRIDS, VISION_WIDTH, 1_000 + seed)[0]}


def merger_draw(seed: int) -> dict[str, np.ndarray]:
    """A seed's trunk features of MIMO_GRIDS' two rows, the merger's input, padded into one batch."""
    rng = np.random.default_rng([0x3e9, seed])
    rows = [rng.standard_normal((int(np.prod(grid)), MIMO_VISION["hidden_size"])).astype(np.float32)
            for grid in MIMO_GRIDS]
    return {"trunk": padded(rows).astype(np.float32)}


def record_mimo_vision() -> None:
    """The fixture's tower and merger (mimo-vision-tiny's tensors, loaded into the release's modules) over
    seeds: the trunk of drawn pixels and the merger of drawn trunk features, each row's real entries in
    float32 and float64, recorded for check 1 (`record_case`)."""
    from recorded_references import provenance, record_case, save
    from safetensors.torch import load_file
    from torch_precision import float64

    modeling = mimo_module()
    config = SimpleNamespace(**MIMO_VISION)
    directory = FIXTURES / "mimo-vision-tiny"
    held = {name: tensor for file in ("model.safetensors", "projector.safetensors")
            for name, tensor in load_file(str(directory / file)).items()}
    narrow = modeling.MiMoVisionTransformer(config).float().eval()
    # The merger's biases zero, as the release stores none and the fixture holds none.
    with torch.no_grad():
        for name, tensor in narrow.merger.named_parameters():
            if name.endswith("bias"):
                tensor.zero_()
    narrow.load_state_dict({**narrow.state_dict(), **held})
    with float64():
        wide = modeling.MiMoVisionTransformer(config).eval()
        wide.load_state_dict({name: tensor.double() for name, tensor in narrow.state_dict().items()})

    def trunks(model, pixels, dtype):
        caught, out = {}, []
        hook = model.merger.register_forward_hook(lambda module, inputs, output: caught.update(x=inputs[0]))
        with torch.no_grad():
            for row, grid in zip(pixels, MIMO_GRIDS, strict=True):
                model(torch.from_numpy(row[:int(np.prod(grid))]).to(dtype), torch.tensor([grid]))
                out.append(caught["x"].double().numpy())
        hook.remove()
        return {"tower": np.concatenate(out)}

    def merged(model, trunk, dtype):
        with torch.no_grad():
            return {"merger": np.concatenate([
                model.merger(torch.from_numpy(row[:int(np.prod(grid))]).to(dtype)).double().numpy()
                for row, grid in zip(trunk, MIMO_GRIDS, strict=True)])}

    def case(run, name):
        """A site's float32 run and float64 twin over a seed's `name` input."""
        def evaluate(inputs):
            narrow_out = run(narrow, inputs[name], torch.float32)
            with float64():
                return narrow_out, run(wide, inputs[name], torch.float64)
        return evaluate

    # Summarized: the test's float64 run is held to the truth's sketch (`indexed_recording`'s `truth_fn`).
    arrays = {**record_case("mimo tower", vision_draw, case(trunks, "pixels"), store_inputs=False,
                            summary=True),
              **record_case("mimo merger", merger_draw, case(merged, "trunk"), store_inputs=False,
                            summary=True)}
    written = save(directory / "recording.npz", {**arrays, "provenance": provenance(
        MIMO_REPO, MIMO_REVISION, "cpu",
        "PYTHONPATH=src:tests python tools/mimo_reference.py --record-vision",
        torch=torch.__version__)}, reference=f"{MIMO_REPO}@{MIMO_REVISION}", license="mit")
    print(f"{written}: {sorted(arrays)}")


CODE_LENGTHS = (11, 4)
"""Each drawn clip's code frames, its last group partial."""


def mels_draw(seed: int) -> dict[str, np.ndarray]:
    """A seed's mel frames of the two clips of LENGTHS."""
    rng = np.random.default_rng([0x3ea, seed])
    return {f"mel_{clip}": rng.standard_normal((length, TOKENIZER["n_mels"])).astype(np.float32)
            for clip, length in enumerate(LENGTHS)}


def codes_draw(seed: int) -> dict[str, np.ndarray]:
    """A seed's codes of two clips of CODE_LENGTHS, each channel within its codebook."""
    rng = np.random.default_rng([0x3eb, seed])
    sizes = TOKENIZER["codebook_size"]
    return {f"codes_{clip}": np.stack([rng.integers(0, size, length) for size in sizes],
                                      axis=-1).astype(np.int64)
            for clip, length in enumerate(CODE_LENGTHS)}


def record_mimo_audio() -> None:
    """The fixture's tokenizer encoder and audio encoder (mimo-audio-tiny's tensors in the release's
    modules) over seeds: the pooled features of drawn mels, the reference's batches whole and bounded at
    30 frames, and each drawn clip's encoded groups, in float32 and float64 (`record_case`, summarized:
    the test's float64 runs are held to the truths' sketches)."""
    from recorded_references import provenance, record_case, save
    from torch_precision import float64

    modeling = mimo_module()
    with np.load(AUDIO_FIXTURE / "model.npz") as stored:
        held = {name: torch.from_numpy(np.asarray(stored[name])) for name in stored.files
                if name.startswith(("tokenizer/", "encoder/"))}

    def build(dtype):
        modules = audio_modules(modeling)
        prefixes = ("tokenizer/", "encoder/audio_encoder.", "encoder/speech_embeddings.")
        for module, prefix in zip(modules, prefixes, strict=True):
            module.load_state_dict({name.removeprefix(prefix): tensor for name, tensor in held.items()
                                    if name.startswith(prefix)}, strict=False)
        return tuple(module.to(dtype) for module in modules)

    narrow = build(torch.float32)
    with float64():
        wide = build(torch.float64)

    def features(modules, inputs, dtype, bound):
        mels = [inputs[f"mel_{clip}"] for clip in range(len(LENGTHS))]
        return {"features": audio_tokenized(modeling, modules, mels, dtype, bound)[0].astype(np.float64)}

    def encoded(modules, inputs, dtype):
        codes = [inputs[f"codes_{clip}"] for clip in range(len(CODE_LENGTHS))]
        return {"encoded": np.concatenate(audio_encoded(modeling, modules, codes)).astype(np.float64)}

    def both(run, **options):
        def evaluate(inputs):
            reference = run(narrow, inputs, torch.float32, **options)
            with float64():
                return reference, run(wide, inputs, torch.float64, **options)
        return evaluate

    arrays = {**record_case("tokenizer features", mels_draw, both(features, bound=256000),
                            store_inputs=False, summary=True),
              **record_case("bounded tokenizer features", mels_draw, both(features, bound=30),
                            store_inputs=False, summary=True),
              **record_case("audio encoder", codes_draw, both(encoded), store_inputs=False, summary=True)}
    written = save(AUDIO_FIXTURE / "recording.npz", {**arrays, "provenance": provenance(
        MIMO_REPO, MIMO_REVISION, "cpu",
        "PYTHONPATH=src:tests python tools/mimo_reference.py --record-audio", torch=torch.__version__)},
        reference=f"{MIMO_REPO}@{MIMO_REVISION}", license="mit")
    print(f"{written}: {sorted(arrays)}")


def write_mel() -> None:
    """torchaudio's mel frames of a fixed half second at MiMo's settings, fp32 and float64."""
    from torchaudio.transforms import MelSpectrogram

    waveform = np.random.RandomState(5).randn(12000).astype(np.float32) * 0.3

    from torch_precision import default_dtype

    def mel(dtype):
        # The filters and the window are built in the default dtype, so float64's are built in float64.
        with default_dtype(dtype):
            transform = MelSpectrogram(sample_rate=24000, n_fft=960, hop_length=240, win_length=960, f_min=0,
                                       f_max=None, n_mels=128, power=1.0, center=True)
        spec = transform(torch.from_numpy(waveform).to(dtype)[None, :])
        return torch.log(torch.clip(spec, min=1e-7)).squeeze(0).T.numpy()

    AUDIO_FIXTURE.mkdir(parents=True, exist_ok=True)
    np.savez(AUDIO_FIXTURE / "mel.npz", precision="ieee", waveform=waveform, mel=mel(torch.float32),
             mel_f64=mel(torch.float64))


def audio_modules(modeling):
    """The release's tokenizer encoder, audio encoder and speech embeddings at the fixture's sizes."""
    tokenizer = modeling.AudioTokenizerEncoder(modeling.MiMoAudioTokenizerConfig(**TOKENIZER)).eval()
    config = SimpleNamespace(**AUDIO)
    return tokenizer, modeling.MiMoAudioEncoder(config).eval(), modeling._build_speech_embeddings(config)


def audio_encoded(modeling, modules, codes):
    """The audio encoder over each clip's codes, a clip's groups apart."""
    _, encode, embed = modules
    with torch.no_grad():
        grouped = [modeling._pad_and_group_audio_codes(torch.as_tensor(c), AUDIO["audio_channels"],
                                                       AUDIO["group_size"]) for c in codes]
        hidden = encode._apply_input_local_transformer(encode._apply_speech_embeddings(torch.cat(grouped),
                                                                                        embed))
        out = encode.projection(hidden.reshape(hidden.shape[0], -1)).numpy()
    return np.split(out, np.cumsum([len(g) for g in grouped])[:-1])


def audio_tokenized(modeling, modules, mels, dtype, bound):
    """The pooled features of every segment in order and each clip's codes, the reference's batches
    bounded at `bound` frames (its `_at_group_by_length`, run under that bound)."""
    grouping = modeling._at_group_by_length
    caught = []
    hook = modules[0].down_sample_norm.register_forward_hook(lambda module, args, out: caught.append(out))
    modeling._at_group_by_length = lambda features, lengths, _: grouping(features, lengths, bound)
    try:
        codes = modeling.tokenize_audio_batch([torch.from_numpy(m).to(dtype) for m in mels], modules[0],
                                              segment_size=AUDIO["audio_segment_size"])
    finally:
        modeling._at_group_by_length = grouping
        hook.remove()
    return torch.cat(caught).numpy(), [c.numpy() for c in codes]


def write_mimo_audio_tiny() -> None:
    from torch_precision import float64

    modeling = mimo_module()
    torch.manual_seed(1234)
    tokenizer, encoder, embeddings = audio_modules(modeling)
    for index, module in enumerate((tokenizer, encoder, embeddings)):
        scatter(module, 1234 + index)
    mels = [np.random.RandomState(7 + i).randn(length, TOKENIZER["n_mels"]).astype(np.float32)
            for i, length in enumerate(LENGTHS)]

    def encoded(modules, codes):
        return audio_encoded(modeling, modules, codes)

    def tokenized(modules, dtype, bound):
        return audio_tokenized(modeling, modules, mels, dtype, bound)

    modules = (tokenizer, encoder, embeddings)
    arrays = {}
    for prefix, bound in (("", 256000), ("bounded_", 30)):
        features, codes = tokenized(modules, torch.float32, bound)
        arrays.update({f"{prefix}features": features, **{f"{prefix}codes_{i}": c for i, c in enumerate(codes)},
                       **{f"{prefix}embedded_{i}": e for i, e in enumerate(encoded(modules, codes))}})
    with float64():
        wide = audio_modules(modeling)
        for source, target in zip(modules, wide, strict=True):
            target.load_state_dict({name: tensor.double() if tensor.is_floating_point() else tensor
                                    for name, tensor in source.state_dict().items()})
        for prefix, bound in (("", 256000), ("bounded_", 30)):
            features, codes = tokenized(wide, torch.float64, bound)
            assert all(np.array_equal(a, arrays[f"{prefix}codes_{i}"]) for i, a in enumerate(codes)), (
                "a code flips in float64")
            # The encoder in float64 over the fp32 run's codes, the codes it reads.
            fp32_codes = [arrays[f"{prefix}codes_{i}"] for i in range(len(LENGTHS))]
            arrays.update({f"{prefix}features_f64": features,
                           **{f"{prefix}embedded_f64_{i}": e for i, e in enumerate(encoded(wide, fp32_codes))}})
    tensors = {**{f"tokenizer/{name}": tensor for name, tensor in tokenizer.state_dict().items()},
               **{f"encoder/audio_encoder.{name}": tensor for name, tensor in encoder.state_dict().items()
                  if "embed_tokens" not in name and "inv_freq" not in name},
               **{f"encoder/speech_embeddings.{name}": tensor for name, tensor in embeddings.state_dict().items()}}
    AUDIO_FIXTURE.mkdir(parents=True, exist_ok=True)
    width = max(len(m) for m in mels)
    np.savez(AUDIO_FIXTURE / "model.npz", precision="ieee", **{name: t.numpy() for name, t in tensors.items()},
             mels=np.stack([np.pad(m, ((0, width - len(m)), (0, 0))) for m in mels]),
             lengths=np.asarray(LENGTHS, np.int32), **arrays)
    (AUDIO_FIXTURE / "config.json").write_text(json.dumps({"audio_config": AUDIO, "tokenizer": TOKENIZER},
                                                          indent=1) + "\n")
    print(AUDIO_FIXTURE, sorted(p.name for p in AUDIO_FIXTURE.iterdir()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mel", action="store_true", help="write mimo-audio-tiny/mel.npz, under torchaudio")
    parser.add_argument("--record-vision", action="store_true",
                        help="record mimo-vision-tiny's tower and merger over seeds")
    parser.add_argument("--record-audio", action="store_true",
                        help="record mimo-audio-tiny's tokenizer and audio encoder over seeds")
    arguments = parser.parse_args()
    if arguments.record_audio:
        record_mimo_audio()
        return
    if arguments.mel:
        write_mel()
        return
    if arguments.record_vision:
        record_mimo_vision()
        return
    write_mimo_vision_tiny()
    write_mimo_audio_tiny()


if __name__ == "__main__":
    main()
