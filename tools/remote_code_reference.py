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

`--padding` records instead the rows each release's own token lookup leaves
untrained, under the fixture's config with its pad id stated, left out and
null (`reference_source.record_padding`), as padding.json beside it. It reads
the release's code and config alone, under each release's transformers as
above:
  /tmp/remote/bin/python tools/remote_code_reference.py --padding llada-tiny dream-tiny
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
from reference_source import config_lookups, record_padding
from safetensors.torch import load_file, save_file
from torch_precision import float64, float64_twin
from transformers import AutoConfig

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "hf"
RELEASES = {
    "llada-tiny": ("GSAI-ML/LLaDA-8B-Base", "0f2787f2d87eac5eed8a087d5ecd24277e6255b2",
                   ("configuration_llada.py", "modeling_llada.py"), "AutoModel"),
    "dream-tiny": ("Dream-org/Dream-v0-Base-7B", "6572adb5535263e4d1a337b56942ba48b6dee2a9",
                   ("configuration_dream.py", "modeling_dream.py", "generation_utils.py"),
                   "AutoModel"),
    "mimo-v2-tiny": ("XiaomiMiMo/MiMo-V2.6-Pro-MOPD", "6a7f202e8dbd0e57b3fe1da03917d08d75abc4b8",
                     ("configuration_mimo_v2.py", "modeling_mimo_v2.py"), "AutoModelForCausalLM"),
    # modeling_lfm3.py imports its Triton conv from lfm3_conv.py, which its CPU run passes by.
    "lfm3-tiny": ("LiquidAI/LFM3-1.7B", "40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd",
                  ("configuration_lfm3.py", "modeling_lfm3.py", "lfm3_conv.py"), "AutoModelForCausalLM"),
}
"""Each fixture's release: the Hub repo, the commit its code is read at, the
files that code is, and the auto class its language model registers as. Each
release's float64 run is built and run under `torch_precision.float64`, which
widens the pins its code carries (MiMo's router casts its logits with
`.type(torch.float32)`) and computes the tables its constructor builds (a
rotary `inv_freq`) in float64."""


def released_classes(name: str, workspace: Path):
    """The release's own config of a dict (its code's files at its commit, beside the dict with the
    release's `auto_map`, in `workspace`) and its model class over such a config."""
    repo, revision, files, auto = RELEASES[name]
    directory = workspace / name
    directory.mkdir(parents=True)
    for file in files:
        shutil.copy(hf_hub_download(repo, file, revision=revision), directory / file)
    released = json.loads(Path(hf_hub_download(repo, "config.json", revision=revision)).read_text())

    def configured(config: dict):
        (directory / "config.json").write_text(json.dumps({**config, "auto_map": released["auto_map"]}))
        return AutoConfig.from_pretrained(str(directory), trust_remote_code=True)

    def model(config) -> torch.nn.Module:
        return getattr(transformers, auto).from_config(config, trust_remote_code=True,
                                                       attn_implementation="eager")
    return configured, model


def released_model(name: str, scratch: Path, draw: bool = False) -> torch.nn.Module:
    """The release's own model class over the fixture's config and weights,
    or, with `draw`, weights it draws and writes as the fixture's."""
    configured, built = released_classes(name, scratch)
    model = built(configured(json.loads((FIXTURES / name / "config.json").read_text())))
    if draw:
        torch.manual_seed(0)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.copy_(torch.randn_like(parameter) * 0.1)
        save_file({key: value.contiguous() for key, value in model.state_dict().items()},
                  str(FIXTURES / name / "model.safetensors"))
        vocabulary = json.loads((FIXTURES / name / "config.json").read_text())["vocab_size"]
        ids = torch.randint(0, vocabulary, (2, 16)).numpy().astype(np.int32)
        np.save(FIXTURES / name / "input_ids.npy", ids)
    model.load_state_dict(load_file(str(FIXTURES / name / "model.safetensors")), strict=True)
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", default=sorted(RELEASES))
    parser.add_argument("--check", action="store_true", help="compare with logits.npy instead of writing")
    parser.add_argument("--draw", action="store_true", help="draw the fixture's weights and ids first")
    parser.add_argument("--padding", action="store_true", help="record the rows each lookup leaves untrained")
    arguments = parser.parse_args()
    with tempfile.TemporaryDirectory() as scratch:
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
                wide = float64_twin(model, lambda: released_model(name, Path(scratch) / "f64"))
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
