#!/usr/bin/env python3
"""Write tiny masked-decoder and MiMo logits with the pinned releases' own model code.

LLaDA and Dream ship no transformers class: their Hub repos carry their own
configuration and modeling files, loaded with trust_remote_code. This fetches
those files at a pinned commit, builds the model from the fixture's tiny
config (the released config's auto_map added, so transformers finds the
classes), loads the fixture's committed weights and runs its committed token
ids, in fp32 for logits.npy and in float64 for logits_f64.npy, the exact
value the tests measure both fp32 runs from (tests/reference_error.py).

The weights and ids are inputs, read here as they are: drawn by
tools/hf_reference.py's write_diffusion_tiny as of commit 18511496038f
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

MiMo-V2.6's code reads transformers 5's masking and rotary utilities (its
config records 5.3.0), so mimo-v2-tiny runs under Dew's locked environment
(transformers 5.16.1), with --draw the first time: the release's own class
draws the fixture's weights (every parameter, the sinks and the balancing
bias too, from N(0, 0.1^2) at seed 0, as the class leaves the sinks empty)
and its token ids, written in the class's split layout:
  python tools/remote_code_reference.py mimo-v2-tiny --draw

The LLaDA MoE tinies use --draw too. LLaDA-MoE, v2 and LLaDA2.0 run
under transformers 4.57.1; LLaDA2.1 and 2.2 under 5.16.1. Forty token
grids hold the native fp32 forward to its float64 execution. LLaDA2's
probes carry the release's explicit block-causal attention mask; this
records forward parity, not parity with its block-wise generation recipe.
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import transformers
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file
from transformers import AutoConfig

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "hf"
RELEASES = {
    "llada-tiny": ("GSAI-ML/LLaDA-8B-Base", "0f2787f2d87eac5eed8a087d5ecd24277e6255b2",
                   ("configuration_llada.py", "modeling_llada.py"), "AutoModel", False),
    "dream-tiny": ("Dream-org/Dream-v0-Base-7B", "6572adb5535263e4d1a337b56942ba48b6dee2a9",
                   ("configuration_dream.py", "modeling_dream.py", "generation_utils.py"),
                   "AutoModel", False),
    "mimo-v2-tiny": ("XiaomiMiMo/MiMo-V2.6-Pro-MOPD", "6a7f202e8dbd0e57b3fe1da03917d08d75abc4b8",
                     ("configuration_mimo_v2.py", "modeling_mimo_v2.py"), "AutoModelForCausalLM", True),
    "llada-moe-tiny": ("inclusionAI/LLaDA-MoE-7B-A1B-Base", "daccaf1ccbf263a427bc9ac2a23b0b004fb275bf",
                       ("configuration_lladamoe.py", "modeling_lladamoe.py"), "AutoModelForCausalLM", True),
    "llada-moe-v2-tiny": ("GSAI-ML/LLaDA-MoE-v2-30B-A3B-Base", "2a513f7a4634e426b2ab030e6defb24001d0c847",
                          ("configuration_lladamoe_v2.py", "modeling_lladamoe_v2.py"),
                          "AutoModelForCausalLM", True),
    "llada2-tiny": ("inclusionAI/LLaDA2.0-mini", "dad945cac317da394b390f82c7b40691d8a881ed",
                    ("configuration_llada2_moe.py", "modeling_llada2_moe.py"), "AutoModelForCausalLM", True),
    "llada21-tiny": ("inclusionAI/LLaDA2.1-mini", "20e64e2ad21644d0e5248586ed9c942cdd45de0f",
                     ("configuration_llada2_moe.py", "modeling_llada2_moe.py"), "AutoModelForCausalLM", True),
    "llada22-tiny": ("inclusionAI/LLaDA2.2-flash", "9b78da3f9149f1e239fb6d15f20a326fde574b9f",
                     ("configuration_llada2_moe.py", "modeling_llada2_moe.py"), "AutoModelForCausalLM", True),
}
"""Each fixture's release: the Hub repo, the commit its code is read at, the
files that code is, the auto class its language model registers as, and
whether the code pins float32 (MiMo's router casts its logits with
`.type(torch.float32)`), so its float64 run is built and run under
`diffusers_wan_reference.float64`, which widens those pins and the rotary
tables built at init."""


MOE = ("llada-moe-tiny", "llada-moe-v2-tiny", "llada2-tiny", "llada21-tiny", "llada22-tiny")


def released_model(name: str, scratch: Path, draw: bool = False) -> torch.nn.Module:
    """The release's own model class over the fixture's config and weights,
    or, with `draw`, weights it draws and writes as the fixture's."""
    repo, revision, files, auto, _ = RELEASES[name]
    directory = scratch / name
    directory.mkdir(parents=True)
    for file in files:
        shutil.copy(hf_hub_download(repo, file, revision=revision), directory / file)
    released = json.loads(Path(hf_hub_download(repo, "config.json", revision=revision)).read_text())
    config = json.loads((FIXTURES / name / "config.json").read_text())
    (directory / "config.json").write_text(json.dumps({**config, "auto_map": released["auto_map"]}))
    reader = getattr(transformers, auto)
    model = reader.from_config(AutoConfig.from_pretrained(str(directory), trust_remote_code=True),
                               trust_remote_code=True, attn_implementation="eager")
    if draw:
        torch.manual_seed(0)
        with torch.no_grad():
            for key, parameter in model.named_parameters():
                noise = torch.randn_like(parameter) * (0.2 if name in MOE else 0.1)
                parameter.copy_(noise + 1 if name in MOE and 'norm' in key else noise)
            for key, buffer in model.named_buffers():
                if name in MOE and key.endswith('expert_bias'):
                    buffer.copy_(torch.randn_like(buffer) * 0.3)
        save_file({key: value.contiguous() for key, value in model.state_dict().items()},
                  str(FIXTURES / name / "model.safetensors"))
        vocabulary = json.loads((FIXTURES / name / "config.json").read_text())["vocab_size"]
        shape = (40, 2, 16) if name in MOE else (2, 16)
        ids = torch.randint(0, vocabulary, shape).numpy().astype(np.int32)
        if name in MOE:
            np.save(FIXTURES / name / "probe_ids.npy", ids)
        np.save(FIXTURES / name / "input_ids.npy", ids[0] if name in MOE else ids)
    model.load_state_dict(load_file(str(FIXTURES / name / "model.safetensors")), strict=True)
    return model.eval()


def forward(model, ids, name):
    """The release's inference forward, with LLaDA2's block-causal mask."""
    fields = {}
    if name in ('llada2-tiny', 'llada21-tiny', 'llada22-tiny'):
        blocks = torch.arange(ids.shape[-1]) // 4
        visible = blocks[None, :] <= blocks[:, None]
        fields['attention_mask'] = torch.where(visible, 0.0, -torch.inf)[None, None].expand(
            ids.shape[0], 1, -1, -1)
    return model(input_ids=ids, **fields).logits


def ensemble(model, ids, name):
    if name in MOE:
        return np.stack([forward(model, tokens, name).detach().numpy() for tokens in ids])
    return forward(model, ids, name).detach().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", default=sorted(RELEASES))
    parser.add_argument("--check", action="store_true", help="compare with logits.npy instead of writing")
    parser.add_argument("--draw", action="store_true", help="draw the fixture's weights and ids first")
    arguments = parser.parse_args()
    with tempfile.TemporaryDirectory() as scratch:
        for name in arguments.names:
            model = released_model(name, Path(scratch) / "fp32", arguments.draw)
            ids_name = 'probe_ids.npy' if name in MOE else 'input_ids.npy'
            ids = torch.from_numpy(np.load(FIXTURES / name / ids_name).astype(np.int64))
            with torch.no_grad():
                fp32 = ensemble(model.float(), ids, name)
                if RELEASES[name][4]:
                    from diffusers_wan_reference import float64

                    with float64():
                        wide = released_model(name, Path(scratch) / "f64").double()
                        f64 = ensemble(wide, ids, name)
                else:
                    f64 = ensemble(model.double(), ids, name)
            if arguments.check:
                saved = 'probe_logits.npy' if name in MOE else 'logits.npy'
                same = np.array_equal(np.load(FIXTURES / name / saved), fp32)
                print(name, saved, 'bitwise' if same else 'differs')
            else:
                if name in MOE:
                    np.save(FIXTURES / name / 'probe_logits.npy', fp32)
                    np.save(FIXTURES / name / 'probe_logits_f64.npy', f64)
                np.save(FIXTURES / name / "logits.npy", fp32[0] if name in MOE else fp32)
                np.save(FIXTURES / name / "logits_f64.npy", f64[0] if name in MOE else f64)
                print(name, fp32.shape)


if __name__ == "__main__":
    main()
