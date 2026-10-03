#!/usr/bin/env python3
"""Write a tiny decoder fixture's logits with transformers, from the
fixture's own checkpoint.

gemma4-ple, gemma4-kvshare and gemma4-e2b hold a random-weight Gemma 4 text
checkpoint (config.json, model.safetensors), the token ids it is run on
(input_ids.npy) and the logits transformers computes for them (logits.npy).
This reads the first three and writes the fourth: AutoModelForCausalLM at
fp32 in eval mode, with transformers' default SDPA attention, which is what
the committed logits were computed with (`--check` reproduces them bit for
bit; eager attention differs by up to 1.6e-6). `--check` compares instead of
writing.

Run with transformers 5.16.1 and Torch 2.14.0 CPU (each fixture's meta.json
records both):
  python tools/hf_checkpoint_logits.py tests/fixtures/hf/gemma4-ple \
      tests/fixtures/hf/gemma4-kvshare tests/fixtures/hf/gemma4-e2b
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM


def logits(directory: Path) -> np.ndarray:
    model = AutoModelForCausalLM.from_pretrained(str(directory), dtype=torch.float32,
                                                 attn_implementation="sdpa").eval()
    ids = torch.from_numpy(np.load(directory / "input_ids.npy").astype(np.int64))
    with torch.no_grad():
        return model(input_ids=ids).logits.to(torch.float32).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--check", action="store_true", help="compare with logits.npy instead of writing it")
    arguments = parser.parse_args()
    for directory in arguments.directories:
        values = logits(directory)
        if arguments.check:
            committed = np.load(directory / "logits.npy")
            same = np.array_equal(committed, values)
            difference = np.max(np.abs(committed - values))
            print(directory.name, "bitwise" if same else f"max |difference| {difference:.3g}")
        else:
            np.save(directory / "logits.npy", values)
            print(directory.name, values.shape)


if __name__ == "__main__":
    main()
