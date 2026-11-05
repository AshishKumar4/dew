#!/usr/bin/env python3
"""Write tests/fixtures/decoding/penalties.npz: vLLM's frequency and
presence penalties on logits drawn over check 1's seeds.

vLLM applies them in `apply_penalties` (vllm/model_executor/layers/utils.py,
read at the v0.30.0 commit and run as published with
`get_token_bin_counts_and_mask`). Its repetition penalty is a custom op,
imported inside the function from `vllm._custom_ops`; the module here
holds that file's own `apply_repetition_penalties`, which on a CPU tensor
runs its torch path, `apply_repetition_penalties_torch`, both as published.
The repetition penalty stays 1, so the run is the frequency and presence
penalties alone: OpenAI's `frequency_penalty` times each token's count
among the generated tokens and `presence_penalty` times whether it was
generated at all, the prompt counting for neither.

The rows: four histories over a 17-token vocabulary, prompts of up to five
tokens and outputs of up to four, with tokens that appear in the prompt
alone, in the output alone, in both and several times, padded with the
vocabulary size as vLLM pads them. Each penalty pair is a site of its own,
`case_<index>`, recorded over its seeds (`recorded_references.record_case`):
a seed's float32 logits, standard normal, vLLM's run on them in float32 the
reference and in float64 the truth.

    python tools/vllm_penalty_reference.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch
from recorded_references import record_case
from reference_source import definitions

VLLM = "https://raw.githubusercontent.com/vllm-project/vllm/ced6857afa0ea7b2e3f0846a62e1394e90f15607/"
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "decoding" / "penalties.npz"
VOCAB = 17
PROMPTS = ([3, 3, 9, 1], [7, 2, 5, 2, 7], [4, 11], [16, 0, 8])
OUTPUTS = ([9, 9, 1], [2], [11, 11, 11, 6], [])
PENALTIES = ((0.7, 0.0), (0.0, 0.7), (0.45, -0.3), (-1.2, 1.6))
"""(frequency, presence) pairs; OpenAI's range is -2 to 2."""


def padded(rows: tuple[list[int], ...]) -> torch.Tensor:
    width = max(1, max(len(row) for row in rows))
    return torch.tensor([row + [VOCAB] * (width - len(row)) for row in rows], dtype=torch.long)


def main() -> None:
    ops = definitions(VLLM + "vllm/_custom_ops.py",
                      ("apply_repetition_penalties_torch", "apply_repetition_penalties"), {"torch": torch})
    # apply_repetition_penalties dispatches on logits.is_cuda; its CUDA
    # branch is never taken on these CPU tensors.
    ops["apply_repetition_penalties_cuda"] = None
    sys.modules["vllm"] = types.ModuleType("vllm")
    sys.modules["vllm._custom_ops"] = types.SimpleNamespace(**ops)
    published = definitions(VLLM + "vllm/model_executor/layers/utils.py",
                            ("get_token_bin_counts_and_mask", "apply_penalties"), {"torch": torch})
    arrays = {"penalties": np.asarray(PENALTIES), "prompt": padded(PROMPTS).numpy(),
              "output": padded(OUTPUTS).numpy()}

    def draw(seed):
        rng = np.random.default_rng([0x9E7, seed])
        return {"logits": rng.standard_normal((len(PROMPTS), VOCAB)).astype(np.float32)}

    for index, (frequency, presence) in enumerate(PENALTIES):
        def scored(logits, dtype, frequency=frequency, presence=presence):
            def column(value):
                return torch.full((len(PROMPTS),), value, dtype=dtype)

            penalized = published["apply_penalties"](torch.tensor(logits, dtype=dtype), padded(PROMPTS),
                                                      padded(OUTPUTS), column(presence), column(frequency),
                                                      column(1.0))
            return {"logits": penalized.numpy()}

        arrays.update(record_case(f"case_{index}", draw, lambda inputs, scored=scored: (
            scored(inputs["logits"], torch.float32), scored(inputs["logits"], torch.float64))))
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(FIXTURE, **arrays)
    print(f"{FIXTURE}: {len(PENALTIES)} penalty pairs over {len(PROMPTS)} rows")


if __name__ == "__main__":
    main()
