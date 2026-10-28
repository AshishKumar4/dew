#!/usr/bin/env python3
"""Write the gemma4-ple, gemma4-kvshare and gemma4-e2b fixtures.

Each holds a tiny Gemma 4 text checkpoint (config.json, model.safetensors),
the token ids it is run on (input_ids.npy) and the logits transformers
computes for them (logits.npy). The weights are transformers' own
initialization under `torch.manual_seed` at the seed meta.json records, the
ids are the two fixed rows below, and the logits come from
AutoModelForCausalLM at fp32 in eval mode with transformers' default SDPA
attention (eager attention differs by up to 1.6e-6). config.json and
meta.json's seed are the inputs; a run records the Torch and transformers
releases it ran under in meta.json, and the transformers release in its
tests/fixtures/provenance.json entries. `--check` compares instead of
writing, and reproduces every committed file's values bit for bit.

Run with the transformers release the test suite pins, on Torch CPU:
  python tools/gemma4_tiny_reference.py tests/fixtures/hf/gemma4-ple \\
      tests/fixtures/hf/gemma4-kvshare tests/fixtures/hf/gemma4-e2b
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import transformers
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


def _object_span(text: str, start: int) -> tuple[int, int]:
    """The [start, end) span of the JSON object opening at `start`."""
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return start, index + 1
    raise ValueError("unterminated JSON object")


def record_provenance(directories: list[Path]) -> None:
    """This run's transformers release in each written fixture's record in
    tests/fixtures/provenance.json, every other byte of the file as it was."""
    path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "provenance.json"
    text = path.read_text()
    records = json.loads(text)
    for directory in directories:
        key = f"hf/{directory.name}/logits.npy"
        sources = records["references"][key]["sources"]
        if not any(source["repo"] == "huggingface/transformers" for source in sources):
            raise ValueError(f"{key} records no huggingface/transformers source")
        start, end = _object_span(text, text.index("{", text.index(f'"{key}":')))
        block = text[start:end]
        repo = block.index('"repo": "huggingface/transformers"')
        version = block.index('"version": "', repo) + len('"version": "')
        block = block[:version] + transformers.__version__ + block[block.index('"', version):]
        text = text[:start] + block + text[end:]
    json.loads(text)
    path.write_text(text)


def main() -> None:
    from reference_releases import require

    require("transformers")
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
            meta = json.loads((directory / "meta.json").read_text())
            meta["torch"] = torch.__version__
            meta["transformers"] = transformers.__version__
            (directory / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
            print(directory.name, logits.shape)
    if not arguments.check:
        record_provenance(arguments.directories)


if __name__ == "__main__":
    main()
