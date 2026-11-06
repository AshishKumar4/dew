#!/usr/bin/env python3
"""Write the llada-tiny, dream-tiny and mimo-v2-tiny logits with the released model code.

LLaDA and Dream ship no transformers class: their Hub repos carry their own
configuration and modeling files, loaded with trust_remote_code. This fetches
those files at a pinned commit, builds the model from the fixture's tiny
config (the released config's auto_map added, so transformers finds the
classes), loads the fixture's committed weights and runs its committed token
ids, in fp32 for logits.npy and in float64 for logits_f64.npy, the exact
value the tests measure both fp32 runs from (tests/reference_error.py).

The weights and ids are inputs, read here as they are: drawn by
tools/hf_reference.py's write_diffusion_tiny as of commit ab9e7635af21
(dew.interop.verify's scatter_weights at seed 1234 over a port of each
release's module order, which drawing over the release's own order does
not reproduce, and probe_ids).

The remote code was written against transformers 4.46 (both released
config.json files record 4.46.x, and later releases drop attributes it
reads), so run it there:
  uv venv /tmp/remote --python 3.12
  uv pip install --python /tmp/remote/bin/python torch==2.5.1 \
      --index-url https://download.pytorch.org/whl/cpu
  uv pip install --python /tmp/remote/bin/python transformers==4.46.3 \
      safetensors numpy huggingface_hub
  /tmp/remote/bin/python tools/remote_code_reference.py

MiMo-V2.6's code reads transformers 5's masking and rotary utilities, calling
`create_causal_mask(input_embeds=...)`, which 5.2.0 renamed (its config records
5.3.0, which refuses it), so mimo-v2-tiny runs under transformers 5.1.0, with
--draw the first time: the release's own class
draws the fixture's weights (every parameter, the sinks and the balancing
bias too, from N(0, 0.1^2) at seed 0, as the class leaves the sinks empty)
and its token ids, written in the class's split layout:
  python tools/remote_code_reference.py mimo-v2-tiny --draw

`--padding` records instead the rows each release's own token lookup leaves
untrained, under the fixture's config with its pad id stated, left out and
null (`reference_source.record_padding`), as padding.json beside it. It reads
the release's code and config alone, under each release's transformers as
above:
  /tmp/remote/bin/python tools/remote_code_reference.py --padding llada-tiny dream-tiny

LFM3's code (lfm3-tiny, lfm3-biased-tiny) runs under transformers 5.18.0 on
(tests/fixtures/hf/lfm3-tiny/source.json), and LFM3-VL's (lfm3-vl-tiny), the same LFM3 decoder beside
its vision trunk and projector, under the same (tests/fixtures/hf/lfm3-vl-tiny/source.json).

`--record` records each release over seeds at its sites (tests/remote_code_draws.py's `SITES`,
`record_case(..., summary=True)`), as `released.npz` beside the fixture, which
tests/test_remote_code_releases.py holds Dew to by check 1: Dew's side runs in Dew's environment, and
the release's, served over a pipe (`--serve`), under `--release-python`, its own transformers':
  PYTHONPATH=src python tools/remote_code_reference.py lfm3-tiny lfm3-biased-tiny --record \
      --release-python <a Python with transformers 5.18.0>
  PYTHONPATH=src python tools/remote_code_reference.py mimo-v2-tiny --record --release-python <5.1.0>
  PYTHONPATH=src python tools/remote_code_reference.py dream-tiny llada-tiny --record \
      --release-python <4.46.3>
`--opens <export>` checks a Dew export of the fixture opens through the release's own loader as the
release does (`opens`).
"""

import argparse
import json
import runpy
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import reference_worker
import torch
import transformers
from huggingface_hub import hf_hub_download
from reference_source import config_lookups, record_padding
from safetensors.torch import load_file, save_file
from torch_precision import float64, float64_twin
from transformers import AutoConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
# NumPy alone, as the release's environment holds no JAX.
from remote_code_draws import CHUNKS, PADDED, RATE, SITES, draw, positions, vocabulary

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "hf"
_PINNED = runpy.run_path(str(Path(__file__).resolve().parents[1] / "src" / "dew" / "interop"
                             / "release_pins.py"))
"""The releases whose own code is a Dew family's reference (dew.interop.release_pins), read by path: this
tool runs in the release's own environment, which need not hold Dew's."""

RELEASES = {
    "llada-tiny": ("GSAI-ML/LLaDA-8B-Base", "0f2787f2d87eac5eed8a087d5ecd24277e6255b2",
                   ("configuration_llada.py", "modeling_llada.py"), "AutoModel"),
    "dream-tiny": ("Dream-org/Dream-v0-Base-7B", "6572adb5535263e4d1a337b56942ba48b6dee2a9",
                   ("configuration_dream.py", "modeling_dream.py", "generation_utils.py"),
                   "AutoModel"),
    # The family's own pin (dew.interop.release_pins), as LFM3's.
    "mimo-v2-tiny": (*_PINNED["MIMO_V2"][:2], ("configuration_mimo_v2.py", "modeling_mimo_v2.py"),
                     "AutoModelForCausalLM"),
    # The same release's trunk whose sliding layers attend in a geometry of their own (`swa_*` heads and
    # width), which no published checkpoint states apart (ledger 4290).
    "mimo-v2-sliding-tiny": (*_PINNED["MIMO_V2"][:2], ("configuration_mimo_v2.py", "modeling_mimo_v2.py"),
                             "AutoModelForCausalLM"),
    # The same release whole: its ViT and its audio encoder beside the decoder, the audio tokenizer in its
    # folder (`audio_tokenizer`, `load_audio_tokenizer`).
    "mimo-v2-vl-tiny": (*_PINNED["MIMO_V2"][:2], ("configuration_mimo_v2.py", "modeling_mimo_v2.py"),
                        "AutoModelForCausalLM"),
    # The family's own pin (dew.interop.release_pins). modeling_lfm3.py imports its
    # Triton conv from lfm3_conv.py, which its CPU run passes by.
    "lfm3-tiny": (*_PINNED["LFM3"][:2],
                  ("configuration_lfm3.py", "modeling_lfm3.py", "lfm3_conv.py"), "AutoModelForCausalLM"),
    # The same release with its convolutions' bias (`conv_bias`), which its padding masks around.
    "lfm3-biased-tiny": (*_PINNED["LFM3"][:2], ("configuration_lfm3.py", "modeling_lfm3.py", "lfm3_conv.py"),
                         "AutoModelForCausalLM"),
    # LFM3-VL's wrapper, its LFM3 decoder's code its own copy beside its vision code, and its processor's.
    "lfm3-vl-tiny": (*_PINNED["LFM3_VL"][:2], ("configuration_lfm3_vl.py", "modeling_lfm3_vl.py",
                                               "configuration_lfm3.py", "modeling_lfm3.py", "lfm3_conv.py",
                                               "processing_lfm3_vl.py", "image_processing_lfm3_vl.py",
                                               "video_processing_lfm3_vl.py"),
                     "AutoModelForImageTextToText"),
}
PROCESSORS = {
    # Kimi K2.5's processor (KimiK25Processor and its vision processor), its release's own code, whose
    # public call takes `medias` and returns `grid_thws` (`dew.interop.processors.Processor.media`), its
    # config, and the chat template that writes an image part's placeholder.
    "kimi-k25-tiny": ("moonshotai/Kimi-K2.5", "4d01dfe0332d63057c186e0b262165819efb6611",
                      ("kimi_k25_processor.py", "kimi_k25_vision_processing.py", "media_utils.py",
                       "preprocessor_config.json", "chat_template.jinja")),
}
"""Each fixture whose release a check runs only the processor of: the Hub repo, the commit its files are
read at, and the files, its code beside the config and template that code reads (`vendor`)."""
LICENSES = {"lfm3-tiny": "lfm1.0", "lfm3-biased-tiny": "lfm1.0", "lfm3-vl-tiny": "lfm1.0",
            "kimi-k25-tiny": "modified-mit",
            "mimo-v2-tiny": "mit", "mimo-v2-sliding-tiny": "mit", "mimo-v2-vl-tiny": "mit",
            "dream-tiny": "apache-2.0", "llada-tiny": "mit"}
"""The license of each release a recording (`--record`) runs, as its Hub card states it."""
"""Each fixture's release: the Hub repo, the commit its code is read at, the
files that code is, and the auto class its language model registers as. Each
release's float64 run is built and run under `torch_precision.float64`, which
widens the pins its code carries (MiMo's router casts its logits with
`.type(torch.float32)`) and computes the tables its constructor builds (a
rotary `inv_freq`) in float64."""


def release_file(name: str, file: str) -> Path:
    """One of `name`'s release files at its pinned commit (its code, its `config.json`): the copy
    vendored into the fetched store (`--vendor`, tests/fixtures/hf/<name>/release/<file>), read offline
    and checked by its sha256, where it is pinned; else the Hub's."""
    from fetched_fixtures import PINNED, local

    vendored = FIXTURES / name / "release" / file
    if vendored.with_name(vendored.name + PINNED).is_file():
        return local(vendored)
    repo, revision = (RELEASES[name] if name in RELEASES else PROCESSORS[name])[:2]
    return Path(hf_hub_download(repo, file, revision=revision))


def vendor(name: str) -> None:
    """`name`'s release files, its code (a model's, `RELEASES`, or a processor's, `PROCESSORS`) and its
    `config.json`, pinned into the fetched store from the Hub at the release's commit, with its license
    (`fetched_fixtures.pin_release`), so a check that opens the release reads them offline
    (`release_file`)."""
    from fetched_fixtures import pin_release

    repo, revision, files = RELEASES[name][:3] if name in RELEASES else PROCESSORS[name]
    for file in (*files, "config.json"):
        data = Path(hf_hub_download(repo, file, revision=revision)).read_bytes()
        print(pin_release(FIXTURES / name / "release" / file, data, reference=f"{repo}@{revision}/{file}",
                          license=LICENSES[name]), flush=True)


def released_classes(name: str, workspace: Path):
    """The release's own config of a dict (its code's files at its commit, beside the dict with the
    release's `auto_map`, in `workspace`) and its model class over such a config."""
    _, _, files, auto = RELEASES[name]
    directory = workspace / name
    directory.mkdir(parents=True)
    for file in files:
        shutil.copy(release_file(name, file), directory / file)
    released = json.loads(release_file(name, "config.json").read_text())

    def configured(config: dict):
        (directory / "config.json").write_text(json.dumps({**config, "auto_map": released["auto_map"]}))
        return AutoConfig.from_pretrained(str(directory), trust_remote_code=True)

    def model(config) -> torch.nn.Module:
        return getattr(transformers, auto).from_config(config, trust_remote_code=True,
                                                       attn_implementation="eager")
    return configured, model


OMITTED = {"mimo-v2-vl-tiny": (
    "visual.merger.ln_q.bias", "visual.merger.mlp.0.bias", "visual.merger.mlp.2.bias",
    "audio_encoder.input_local_transformer.embed_tokens.weight")}
"""Each fixture's tensors its release's checkpoint does not store, which its loader initializes as a missing
key: MiMo-V2.6's merger's biases, zeros (transformers' `_init_weights`), and its audio encoder's unused
token table (`_keys_to_ignore_on_load_missing`)."""

PREDICTING = ("mimo-v2-vl-tiny",)
"""The fixtures that carry MiMo-V2's prediction layer (`prediction_layer`), under `model.mtp.layers.0.`, which
the release's own class skips on load."""

AUDIO_TOKENIZER = "audio_tokenizer"
"""The folder a release keeps its audio tokenizer in, its own config and weights, which its model loads
(MiMo-V2.6's `load_audio_tokenizer`)."""


def released_model(name: str, scratch: Path, draw: bool = False) -> torch.nn.Module:
    """The release's own model class over the fixture's config and weights,
    or, with `draw`, weights it draws and writes as the fixture's; with the audio tokenizer in its folder
    (`AUDIO_TOKENIZER`) loaded as the release loads it, in the model's dtype."""
    configured, built = released_classes(name, scratch)
    model = built(configured(json.loads((FIXTURES / name / "config.json").read_text())))
    tokenizer = FIXTURES / name / AUDIO_TOKENIZER
    omitted = OMITTED.get(name, ())
    predicting = name in PREDICTING
    if draw:
        torch.manual_seed(0)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.copy_(torch.randn_like(parameter) * 0.1)
        stored = {key: value for key, value in _stored_once(model.state_dict()).items() if key not in omitted}
        if predicting:
            layer = prediction_layer(model)
            with torch.no_grad():
                for parameter in layer.parameters():
                    parameter.copy_(torch.randn_like(parameter) * 0.1)
            stored.update({PREDICTION_STEM + released_name(key): value.contiguous()
                           for key, value in layer.state_dict().items()})
        save_file(stored, str(FIXTURES / name / "model.safetensors"))
        config = json.loads((FIXTURES / name / "config.json").read_text())
        ids = torch.randint(0, vocabulary(config), (2, 16)).numpy().astype(np.int32)
        np.save(FIXTURES / name / "input_ids.npy", ids)
        if "vision_config" in config:
            media_tokenizer(config, FIXTURES / name)
        if (tokenizer / "config.json").is_file():
            draw_audio_tokenizer(model, tokenizer)
    state = model.state_dict()
    held = load_file(str(FIXTURES / name / "model.safetensors"))
    predicted = {key: held.pop(key) for key in list(held) if key.startswith(PREDICTION_STEM)}
    missing, unexpected = model.load_state_dict(held, strict=False)
    with torch.no_grad():
        for key in omitted:
            # As the release's loader initializes a missing bias; an unused table stays as it is.
            if key.endswith(".bias"):
                state[key].zero_()
    # A tied head is its embedding's tensor, which the file stores once (`_stored_once`).
    shared = set(state) - set(_stored_once(state))
    if unexpected or set(missing) != shared | set(omitted):
        raise ValueError(f"{name}: the file's tensors differ from the model's: missing {missing}, "
                         f"unexpected {unexpected}")
    if predicting:
        # The prediction layer's tensors into the layer composed beside the model, whose own load skips them.
        model.prediction_layer = prediction_layer(model)
        model.prediction_layer.load_state_dict({key: predicted[PREDICTION_STEM + released_name(key)]
                                                for key in model.prediction_layer.state_dict()})
    if (tokenizer / "model.safetensors").is_file():
        model.load_audio_tokenizer(str(tokenizer), device="cpu", dtype=next(model.parameters()).dtype)
    return model.eval()


PREDICTION_STEM = "model.mtp.layers.0."
"""Where a MiMo-V2 checkpoint keeps its one prediction layer (vLLM's `model.mtp.layers.{i}.`)."""


def released_name(key: str) -> str:
    """A tensor of `prediction_layer` under its name past `PREDICTION_STEM`: its block's as vLLM names them,
    the norm before the feed-forward `pre_mlp_layernorm`."""
    return key.removeprefix("block.").replace("post_attention_layernorm", "pre_mlp_layernorm")


def prediction_layer(model: torch.nn.Module) -> torch.nn.Module:
    """MiMo-V2's prediction layer as vLLM composes it (vllm/model_executor/models/mimo_v2_mtp.py at 19aba3e,
    `MiMoV2MTPLayer`), of the release's own modules: the token's embedding and the previous position's final
    state each RMS-normed (`enorm`, `hnorm`), side by side through one bias-free map (`eh_proj`), then one
    decoder layer of the sliding kind's attention (its heads, window, base and sinks) over a dense
    feed-forward at `intermediate_size`, attending causally within its window as the release's sliding
    layers do (`create_sliding_window_causal_mask`, the release's rotary of the sliding kind), and a final
    RMS norm (`final_layernorm`); its logits are the model's own head's."""
    import copy

    modeling = sys.modules[type(model).__module__]
    config = copy.deepcopy(model.config)
    config.hybrid_layer_pattern = [1] * config.num_hidden_layers
    config.moe_layer_freq = [0] * config.num_hidden_layers
    size, eps = config.hidden_size, config.layernorm_epsilon

    class PredictionLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enorm = modeling.MiMoV2RMSNorm(size, eps=eps)
            self.hnorm = modeling.MiMoV2RMSNorm(size, eps=eps)
            self.eh_proj = torch.nn.Linear(2 * size, size, bias=False)
            self.block = modeling.MiMoV2DecoderLayer(config, 0)
            self.final_layernorm = modeling.MiMoV2RMSNorm(size, eps=eps)

        def forward(self, embeds, hidden):
            x = self.eh_proj(torch.cat([self.enorm(embeds), self.hnorm(hidden)], dim=-1))
            places = torch.arange(x.shape[1])
            mask = modeling.create_sliding_window_causal_mask(
                config=config, input_embeds=x, attention_mask=None, cache_position=places,
                past_key_values=None, position_ids=places[None])
            x = self.block(x, attention_mask=mask, position_ids=places[None], cache_position=places,
                           position_embeddings=model.model.swa_rotary_emb(x, places[None]))
            return self.final_layernorm(x)
    return PredictionLayer()


def draw_audio_tokenizer(model: torch.nn.Module, folder: Path) -> None:
    """The audio tokenizer of the release's own class (MiMoAudioTokenizer) over the folder's config, its
    parameters and codebooks drawn as the model's are, written as the folder's weights."""
    modeling = sys.modules[type(model).__module__]
    config = json.loads((folder / "config.json").read_text())
    tokenizer = modeling.MiMoAudioTokenizer(modeling.MiMoAudioTokenizerConfig(**config))
    with torch.no_grad():
        for tensor in [*tokenizer.parameters(),
                       *(buffer for name, buffer in tokenizer.named_buffers() if name.endswith(".embed"))]:
            tensor.copy_(torch.randn_like(tensor) * 0.1)
    save_file(_stored_once(tokenizer.state_dict()), str(folder / "model.safetensors"))


MEDIA_TOKENS = {"image_token_id": "<|image|>", "video_token_id": "<|video|>",
                "vision_start_token_id": "<|vision_start|>", "vision_end_token_id": "<|vision_end|>",
                "audio_token_id": "<|audio|>"}
"""A wrapper fixture's media tokens by the config field placing each, as its release's processor spells them
(Lfm3VlProcessor's defaults, processing_lfm3_vl.py), each where its config places one."""


def media_tokenizer(config: dict, directory: Path) -> None:
    """A tokenizer over a wrapper fixture's vocabulary written beside it: each id a word of its own
    (`w<id>`, split on whitespace), the media tokens at the ids its config places them (`MEDIA_TOKENS`),
    matched whole wherever they sit, as its release's processor writes them unspaced, and its pad, start
    and end tokens."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    named = {config[field]: token for field, token in MEDIA_TOKENS.items() if config.get(field) is not None}
    ends = {"pad_token": config["pad_token_id"], "bos_token": config["bos_token_id"],
            "eos_token": config["eos_token_id"]}
    words = {named.get(index, f"w{index}"): index for index in range(vocabulary(config))}
    tokenizer = Tokenizer(models.WordLevel(words, unk_token=f"w{config['pad_token_id']}"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, additional_special_tokens=list(named.values()),
                                   **{role: f"w{index}" for role, index in ends.items()})
    fast.save_pretrained(str(directory))


def _stored_once(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """`state` with each tensor once, under its first name, as `save_pretrained` writes a tied head:
    the embedding's alone."""
    kept: dict[str, torch.Tensor] = {}
    seen = set()
    for key, value in state.items():
        if value.data_ptr() not in seen:
            seen.add(value.data_ptr())
            kept[key] = value.contiguous()
    return kept


OPENED = """import json, sys, numpy, torch, transformers
loader = getattr(transformers, sys.argv[1])
ids = torch.from_numpy(numpy.load(sys.argv[2]).astype('int64'))
opened = []
for directory in sys.argv[3:]:
    model, info = loader.from_pretrained(directory, trust_remote_code=True, dtype=torch.float32,
                                         output_loading_info=True)
    with torch.no_grad():
        logits = model(input_ids=ids).logits
    report = {key: sorted(value) for key, value in info.items()}
    opened.append((type(model).__name__, report, model.state_dict(), logits))
(name, info, state, logits), (other, other_info, other_state, other_logits) = opened
matched = info == other_info and not any(info.values())
print(json.dumps({"class": name == other, "every weight matched": matched,
                  "state": sorted(state) == sorted(other_state)
                  and all(torch.equal(state[key], other_state[key]) for key in state),
                  "logits bitwise": bool(torch.equal(logits, other_logits))}))
"""
"""A fresh interpreter's offline opens of two directories through their own code (`opens`), compared."""


def opens(name: str, export: Path, scratch: Path) -> dict[str, bool]:
    """Whether `export`, a Dew save of `name`'s fixture (`PretrainedDecoder.from_model(...).save`), opens
    offline through the release's own transformers loader as the release does: one fresh interpreter, the
    hub offline and its module cache empty, opens the export, which reads its own code alone, and the
    release (the fixture's config and weights beside the release's code and auto_map at its commit), and
    finds the same class, every weight matched, the same state tensor for tensor and the same logits bit
    for bit over the fixture's ids (ledger 2038). Run in the release's environment, after Dew's writes
    the export:
      python -c 'from dew.interop import Pretrained, PretrainedDecoder; loaded = Pretrained.load(
          "tests/fixtures/hf/lfm3-tiny"); PretrainedDecoder.from_model(loaded.model, loaded.variables).save(
          "/tmp/lfm3-export")'
      <the release's Python> tools/remote_code_reference.py lfm3-tiny --opens /tmp/lfm3-export
    The release's files are its vendored copies where they are pinned (`release_file`). Returns each
    check by name; tests/test_remote_code_releases.py runs it under the release's transformers
    (`release_env`)."""
    import os

    _, _, files, auto = RELEASES[name]
    release = scratch / "release"
    release.mkdir()
    released = json.loads(release_file(name, "config.json").read_text())
    config = json.loads((FIXTURES / name / "config.json").read_text())
    (release / "config.json").write_text(json.dumps({**config, "auto_map": released["auto_map"]}))
    shutil.copy(FIXTURES / name / "model.safetensors", release / "model.safetensors")
    for file in files:
        shutil.copy(release_file(name, file), release / file)
    # The release's own loader, the auto class its config registers it under.
    run = subprocess.run([sys.executable, "-c", OPENED, auto, str(FIXTURES / name / "input_ids.npy"),
                          str(export), str(release)], check=True, capture_output=True, text=True,
                         env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                              "HF_MODULES_CACHE": str(scratch / "modules")})
    return json.loads(run.stdout.strip().splitlines()[-1])


SPELLED = {"mimo-v2-vl-tiny": {"pixel_values_videos": "video_pixel_values"}}
"""Each release's names for the media fields its forward spells otherwise than its processor's: MiMo-V2.6's
video patches (`MiMoV2ForCausalLM.forward`)."""


def released_outputs(model, site: str, inputs: dict, name: str = "") -> dict[str, np.ndarray]:
    """The release's outputs at a recorded `site` (tests/remote_code_draws.py), as Dew's forward
    gives them (tests/model_support.py's `remote_code_site`), the gradients by the release's own tensor
    names, the media under `name`'s own spellings (`SPELLED`). An update steps `model` and restores its
    weights after."""
    ids = torch.from_numpy(inputs["ids"].astype(np.int64))
    if site == "update":
        weights = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        model.zero_grad()
        with torch.enable_grad():
            logits = model(input_ids=ids).logits
            torch.nn.functional.cross_entropy(logits[:, :-1].flatten(0, 1), ids[:, 1:].flatten()).backward()
        torch.optim.SGD(model.parameters(), lr=RATE).step()
        with torch.no_grad():
            stepped = model(input_ids=ids).logits.numpy()
            model.load_state_dict(weights)
        model.zero_grad()
        return {"logits": stepped}
    if site == "gradients":
        model.zero_grad()
        with torch.enable_grad():
            logits = model(input_ids=ids).logits
            (logits * torch.from_numpy(inputs["cotangent"]).to(logits.dtype)).sum().backward()
        return {name: parameter.grad.detach().numpy() for name, parameter in model.named_parameters()}
    with torch.no_grad():
        if site == "packed":
            return {"logits": model(input_ids=ids, position_ids=torch.from_numpy(
                positions().astype(np.int64))).logits.numpy()}
        if site == "decode":
            cache, steps, start = None, [], 0
            for chunk in CHUNKS:
                out = model(input_ids=ids[:, start:start + chunk], past_key_values=cache, use_cache=True)
                cache, start = out.past_key_values, start + chunk
                steps.append(out.logits)
            return {"logits": torch.cat(steps, dim=1).numpy()}
        if site == "biased":
            logits = model(input_ids=ids, attention_mask=torch.from_numpy(inputs["attention_mask"])).logits
            return {"row0": logits[0].numpy(), "row1": logits[1, PADDED:].numpy()}
        if site in ("image", "video", "frames"):
            # The patch vectors in the model's dtype, as its processor's float32 ones are cast to it.
            dtype = next(model.parameters()).dtype
            spelled = SPELLED.get(name, {})
            media = {spelled.get(field, field): torch.from_numpy(value).to(
                dtype if field.startswith("pixel") else torch.int64)
                for field, value in inputs.items() if field != "ids"}
            return {"logits": model(input_ids=ids, **media).logits.numpy()}
        if site == "mtp":
            # The prediction layer's logits: each position's final state beside the next token's embedding.
            hidden = model.model(input_ids=ids).last_hidden_state
            embeds = model.model.embed_tokens(ids[:, 1:])
            return {"logits": model.lm_head(model.prediction_layer(embeds, hidden[:, :-1])).numpy()}
        if site == "audio":
            # Each clip's valid mel frames through the release's own pipeline (`get_audio_feature`: its
            # tokenizer's codes, then its encoder), in the model's dtype, filling the placeholders.
            dtype = next(model.parameters()).dtype
            clips = zip(inputs["input_features"], inputs["input_features_mask"], strict=True)
            mels = [torch.from_numpy(clip[valid]).to(dtype) for clip, valid in clips]
            embeds = model.audio_encoder.get_audio_feature(mels, model.speech_embeddings,
                                                           model.audio_tokenizer.encoder)
            return {"logits": model(input_ids=ids, audio_embeds=embeds).logits.numpy()}
        return {"logits": model(input_ids=ids).logits.numpy()}


def serve(name: str) -> None:
    """The release's side of a recording (`--record`, over tools/reference_worker.py's frames): its fp32
    outputs and their float64 truth at each site and draw the recorder sends."""
    with tempfile.TemporaryDirectory() as scratch:
        native = released_model(name, Path(scratch) / "fp32").float()
        # Its float64 twin (`float64_twin`): built inside `float64`, its constructor's tables float64's.
        wide = float64_twin(native, lambda: released_model(name, Path(scratch) / "f64"))

        def run(request: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            site = str(request.pop("site"))
            reference = released_outputs(native, site, request, name)
            with float64():
                truth = released_outputs(wide, site, request, name)
            return reference_worker.paired(reference, truth)

        reference_worker.serve(run, frames=True)


def record(name: str, release_python: str) -> None:
    """`name`'s recording over seeds at each of its sites (tests/remote_code_draws.py's `SITES`,
    `record_case(..., summary=True)`), the release served by `release_python` in the environment its code
    runs under, as `released.npz` beside the fixture: what its test holds Dew to by check 1. The
    gradients' leaves are the release's tensors, each read into Dew's parameter it loads as (its
    `WeightLayout.restore`)."""
    # Dew's side, in Dew's environment: the release's runs in `release_python`'s.
    from model_support import flat_tree, remote_code_site
    from recorded_references import provenance, record_case, save

    from dew.interop import Pretrained

    source = Pretrained.load(FIXTURES / name, dtype="float32", attention_impl="reference")
    config = json.loads((FIXTURES / name / "config.json").read_text())
    layouts = {layout.name: layout for layout in source.weight_layouts}
    shapes = {path: leaf.shape for path, leaf in flat_tree(source.variables).items()}
    arrays = {}
    with reference_worker.session([release_python, __file__, name, "--serve"]) as worker:
        for site in SITES[name]:
            def evaluate(inputs, site=site):
                arms = reference_worker.arms(worker.ask({"site": np.asarray(site), **inputs}))
                if site != "gradients":
                    return arms
                return tuple({".".join(layouts[tensor].paths[0][1:]): layouts[tensor].restore(
                    value, shapes[".".join(layouts[tensor].paths[0])]) for tensor, value in arm.items()}
                    for arm in arms)

            arrays.update(record_case(
                f"{name} {site}", lambda seed, site=site: draw(site, seed, config), evaluate,
                remote_code_site(source, site), truth_proxy=remote_code_site(source, site, wide=True),
                store_inputs=False, summary=True))
    repo, revision = RELEASES[name][:2]
    versions = subprocess.run([release_python, "-c", "import torch, transformers; "
                               "print(transformers.__version__, torch.__version__)"],
                              capture_output=True, text=True, check=True).stdout.split()
    arrays[f"{name}.provenance"] = provenance(
        repo, revision, "cpu",
        f"PYTHONPATH=src python tools/remote_code_reference.py {name} --record "
        f"--release-python <a Python with transformers {versions[0]}>",
        transformers=versions[0], torch=versions[1])
    print(save(FIXTURES / name / "released.npz", arrays, reference=repo, license=LICENSES[name]), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", default=sorted(RELEASES))
    parser.add_argument("--check", action="store_true", help="compare with logits.npy instead of writing")
    parser.add_argument("--draw", action="store_true", help="draw the fixture's weights and ids first")
    parser.add_argument("--padding", action="store_true", help="record the rows each lookup leaves untrained")
    parser.add_argument("--opens", type=Path,
                        help="check that a Dew export of the fixture opens as the release does")
    parser.add_argument("--record", action="store_true", help="record the release over seeds at its sites")
    parser.add_argument("--vendor", action="store_true",
                        help="pin the release's files into the fetched store, for checks run offline")
    parser.add_argument("--serve", action="store_true", help="the release's side of --record's pipe")
    parser.add_argument("--release-python", default=sys.executable,
                        help="the Python whose transformers runs the release's code, for --record")
    arguments = parser.parse_args()
    if arguments.serve:
        serve(arguments.names[0])
        return
    if arguments.vendor:
        for name in arguments.names:
            vendor(name)
        return
    if arguments.record:
        for name in arguments.names:
            record(name, arguments.release_python)
        return
    with tempfile.TemporaryDirectory() as scratch:
        if arguments.opens is not None:
            # One export is one fixture's save: it opens beside that fixture's release alone.
            if len(arguments.names) != 1:
                parser.error(f"--opens checks one fixture's export, and {len(arguments.names)} are named")
            checks = opens(arguments.names[0], arguments.opens, Path(scratch))
            print(arguments.names[0], "opens as the release does" if all(checks.values())
                  else f"differs: {checks}")
            if not all(checks.values()):
                raise SystemExit(1)
            return
        if arguments.padding:
            for name in arguments.names:
                record_padding(*config_lookups(*released_classes(name, Path(scratch) / "padding")),
                               json.loads((FIXTURES / name / "config.json").read_text()), FIXTURES / name)
            return
        for name in arguments.names:
            model = released_model(name, Path(scratch) / "fp32", arguments.draw)
            ids = torch.from_numpy(np.load(FIXTURES / name / "input_ids.npy").astype(np.int64))
            with torch.no_grad():
                fp32 = model.float()(input_ids=ids).logits.numpy()
                # Its float64 twin (`float64_twin`), so the rotary tables its constructor computes are
                # float64's (a float32 build cast afterwards keeps float32's, Dream's `inv_freq`), run
                # inside `float64`, so a pin its forward carries widens too.
                wide = float64_twin(model, lambda name=name: released_model(name, Path(scratch) / "f64"))
                with float64():
                    f64 = wide(input_ids=ids).logits.numpy()
            if arguments.check:
                same = np.array_equal(np.load(FIXTURES / name / "logits.npy"), fp32)
                print(name, "logits.npy bitwise" if same else "logits.npy differs")
            else:
                np.save(FIXTURES / name / "logits.npy", fp32)
                np.save(FIXTURES / name / "logits_f64.npy", f64)
                print(name, fp32.shape)


if __name__ == "__main__":
    main()
