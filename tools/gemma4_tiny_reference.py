#!/usr/bin/env python3
"""Write the gemma4-ple, gemma4-kvshare and gemma4-e2b fixtures.

Each holds a tiny Gemma 4 text checkpoint (config.json, model.safetensors),
the token ids it is run on (input_ids.npy) and the logits transformers
computes for them (logits.npy). The weights are transformers' own
initialization under `torch.manual_seed` at the seed meta.json records, the
ids are the two fixed rows below, and the logits come from
AutoModelForCausalLM at fp32 in eval mode with transformers' default SDPA
attention (eager attention differs by up to 1.6e-6). config.json and
meta.json are the inputs; `--check` compares instead of writing, and
reproduces every committed file's values bit for bit.

Run with transformers 5.16.1 and Torch 2.14.0 CPU (each meta.json records
both):
  python tools/gemma4_tiny_reference.py tests/fixtures/hf/gemma4-ple \\
      tests/fixtures/hf/gemma4-kvshare tests/fixtures/hf/gemma4-e2b
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM

IDS = np.array([[5, 17, 3, 42, 9, 28, 51, 12, 33, 8, 60, 21],
                [2, 44, 15, 7, 55, 31, 19, 48, 26, 11, 39, 4]], np.int64)


def fixture(directory: Path) -> tuple[dict[str, torch.Tensor], np.ndarray]:
    """The seeded checkpoint and its logits on `IDS`."""
    torch.manual_seed(json.loads((directory / "meta.json").read_text())["seed"])
    model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(str(directory)),
                                             attn_implementation="sdpa").float().eval()
    with torch.no_grad():
        logits = model(input_ids=torch.from_numpy(IDS)).logits.to(torch.float32).numpy()
    weights = {name: tensor.contiguous() for name, tensor in model.state_dict().items()}
    if model.config.tie_word_embeddings:
        # The tied head is the embedding; the checkpoint carries it once, as save_pretrained does.
        weights.pop("lm_head.weight")
    return weights, logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--check", action="store_true", help="compare with the committed files instead")
    arguments = parser.parse_args()
    for directory in arguments.directories:
        weights, logits = fixture(directory)
        if arguments.check:
            committed = load_file(str(directory / "model.safetensors"))
            same = (committed.keys() == weights.keys()
                    and all(torch.equal(committed[name], weights[name]) for name in committed)
                    and np.array_equal(np.load(directory / "input_ids.npy"), IDS)
                    and np.array_equal(np.load(directory / "logits.npy"), logits))
            print(directory.name, "bitwise" if same else "differs")
        else:
            save_file(weights, directory / "model.safetensors")
            np.save(directory / "input_ids.npy", IDS)
            np.save(directory / "logits.npy", logits)
            print(directory.name, logits.shape)


if __name__ == "__main__":
    main()
