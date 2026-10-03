#!/usr/bin/env python3
"""Write the llada-tiny and dream-tiny logits with the released model code.

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
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "hf"
RELEASES = {
    "llada-tiny": ("GSAI-ML/LLaDA-8B-Base", "0f2787f2d87eac5eed8a087d5ecd24277e6255b2",
                   ("configuration_llada.py", "modeling_llada.py")),
    "dream-tiny": ("Dream-org/Dream-v0-Base-7B", "6572adb5535263e4d1a337b56942ba48b6dee2a9",
                   ("configuration_dream.py", "modeling_dream.py", "generation_utils.py")),
}
"""Each fixture's release: the Hub repo, the commit its code is read at and
the files that code is."""


def released_model(name: str, scratch: Path) -> torch.nn.Module:
    """The release's own model class over the fixture's config and weights."""
    repo, revision, files = RELEASES[name]
    directory = scratch / name
    directory.mkdir()
    for file in files:
        shutil.copy(hf_hub_download(repo, file, revision=revision), directory / file)
    released = json.loads(Path(hf_hub_download(repo, "config.json", revision=revision)).read_text())
    config = json.loads((FIXTURES / name / "config.json").read_text())
    (directory / "config.json").write_text(json.dumps({**config, "auto_map": released["auto_map"]}))
    model = AutoModel.from_config(AutoConfig.from_pretrained(str(directory), trust_remote_code=True),
                                  trust_remote_code=True)
    model.load_state_dict(load_file(str(FIXTURES / name / "model.safetensors")), strict=True)
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", default=sorted(RELEASES))
    parser.add_argument("--check", action="store_true", help="compare with logits.npy instead of writing")
    arguments = parser.parse_args()
    with tempfile.TemporaryDirectory() as scratch:
        for name in arguments.names:
            model = released_model(name, Path(scratch))
            ids = torch.from_numpy(np.load(FIXTURES / name / "input_ids.npy").astype(np.int64))
            with torch.no_grad():
                fp32 = model.float()(input_ids=ids).logits.numpy()
                f64 = model.double()(input_ids=ids).logits.numpy()
            if arguments.check:
                same = np.array_equal(np.load(FIXTURES / name / "logits.npy"), fp32)
                print(name, "logits.npy bitwise" if same else "logits.npy differs")
            else:
                np.save(FIXTURES / name / "logits.npy", fp32)
                np.save(FIXTURES / name / "logits_f64.npy", f64)
                print(name, fp32.shape)


if __name__ == "__main__":
    main()
