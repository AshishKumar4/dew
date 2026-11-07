"""transformers' beam search (tools/beam_reference.py's `searched`, transformers 5.16.1's `_beam_search`) over
drawn prompts, in float32 and float64: the reference arm of tools/venv_reference.py's `beam-<name>` sites (its
protocol tools/reference_worker.py), on the CPU.

    python tools/beam_worker.py beam-NAME    `IN OUT` lines on stdin, each `done`

IN holds each draw's `<draw>.ids` `[prompts, length]`, unpadded. OUT gets, for each of beam_reference.py's
`SETTINGS` (`case_<i>`), `<draw>.reference.case_<i>` and `<draw>.truth.case_<i>`: every prompt's returned
beams' scores (`sequences_scores`), in float32 and in float64 (the float64 twin built and run inside
`torch_precision.float64`), under the checkpoint's recorded EOS and pad (its `beam.npz`).
"""

import sys

import numpy as np
import torch
from beam_reference import SETTINGS, searched
from numerics_reference import FIXTURES, decoder
from reference_worker import drawn, serve
from torch_precision import float64, float64_twin


def run(name: str, cases: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Every draw's beams' scores under each setting, in float32 and float64."""
    model = decoder(name, torch.float32)
    wide = float64_twin(model, lambda: decoder(name, torch.float64))
    for each in (model, wide):
        each.generation_config.eos_token_id = None
    with np.load(FIXTURES / name / "beam.npz") as stored:
        eos, pad = int(stored["eos"]), int(stored["pad"])
    def answer(inputs):
        ids = np.asarray(inputs["ids"])
        mask = np.ones_like(ids, bool)
        reference, truth = {}, {}
        for index, (penalty, early) in enumerate(SETTINGS):
            _, scores = searched(model, ids, mask, eos, pad, penalty, early)
            reference[f"case_{index}"] = scores.astype(np.float32)
            with float64():
                truth[f"case_{index}"] = searched(wide, ids, mask, eos, pad, penalty, early)[1]
        return reference, truth

    return drawn(cases, answer)


if __name__ == "__main__":
    if len(sys.argv) != 2 or not sys.argv[1].startswith("beam-"):
        raise SystemExit(__doc__)
    torch.set_num_threads(1)
    serve(lambda cases: run(sys.argv[1].removeprefix("beam-"), cases))
