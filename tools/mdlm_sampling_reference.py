#!/usr/bin/env python3
"""MDLM's sampler continuing a prompt, for tests/fixtures/mdlm/continuation.npz.

The reference is `Diffusion._sample` of kuleshov-group/mdlm (diffusion.py,
at tools/mdlm_reference.py's commit) with the `_ddpm_update`, `forward` and
`_subs_parameterization` it runs, read out of the published file and
executed as written on the stand-in of configs/config.yaml's settings
(tools/mdlm_reference.py), sampling with `ddpm` for `STEPS` steps and
removing the noise left at the end. MDLM samples from an all-masked prior;
a continuation starts from the prompt followed by the masked response
(`_sample_prior` returns that row, the one change), and MDLM's update keeps
every unmasked token, so the prompt stays as given.

The backbone is a stand-in that reads the whole row: every position's logits
are the mean of the row's token embeddings, plus its own position's, times
a head, so the order tokens are revealed in shapes the outcome. What lands:
the prompt, the response length, the weights, the step count, the number
of rows sampled, and the count of each final response, indexed as a number
in base VOCAB - 1 (the mask never remains).

    PYTHONPATH=src python tools/mdlm_sampling_reference.py
"""

import types
from pathlib import Path

import mdlm_reference
import numpy as np
import torch

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "mdlm" / "continuation.npz"
VOCAB, FEATURES, STEPS, ROWS = 5, 6, 3, 200_000
MASK = VOCAB - 1
PROMPT = np.asarray([2, 0])
RESPONSE = 3


class Reader(torch.nn.Module):
    """Logits from the mean token embedding of the row plus each position's own."""

    def __init__(self, embed: np.ndarray, position: np.ndarray, head: np.ndarray):
        super().__init__()
        self.embed, self.position, self.head = (torch.tensor(array, dtype=torch.float64)
                                                for array in (embed, position, head))

    def forward(self, x, sigma):
        context = self.embed[x].mean(dim=1, keepdim=True)
        return (context + self.position) @ self.head


def main() -> None:
    reference = mdlm_reference.published({"_sample", "_ddpm_update", "forward", "_process_sigma",
                                          "_subs_parameterization"})
    generator = np.random.default_rng(4)
    length = len(PROMPT) + RESPONSE
    # The float32 values Dew reads, widened.
    arrays = {name: value.astype(np.float32) for name, value in (
        ("embed", generator.standard_normal((VOCAB, FEATURES))),
        ("position", generator.standard_normal((length, FEATURES))),
        ("head", 0.6 * generator.standard_normal((FEATURES, VOCAB))))}
    model = mdlm_reference.stand_in(reference, Reader(**arrays))
    model.mask_index = MASK
    model.sampler, model.device = "ddpm", "cpu"
    model.config = types.SimpleNamespace(
        model=types.SimpleNamespace(length=length), loader=types.SimpleNamespace(eval_batch_size=ROWS),
        sampling=types.SimpleNamespace(steps=STEPS, noise_removal=True))
    start = torch.tensor(np.concatenate([PROMPT, np.full(RESPONSE, MASK)]))
    model._sample_prior = lambda rows, _length: start.repeat(rows, 1)
    torch.manual_seed(0)
    with torch.no_grad():
        rows = model._sample().numpy()
    assert (rows[:, :len(PROMPT)] == PROMPT).all() and (rows != MASK).all()
    index = rows[:, len(PROMPT):] @ (MASK ** np.arange(RESPONSE - 1, -1, -1))
    counts = np.bincount(index, minlength=MASK ** RESPONSE)
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(FIXTURE, prompt=PROMPT, response=np.asarray(RESPONSE), steps=np.asarray(STEPS),
             rows=np.asarray(ROWS), counts=counts, commit=np.array(mdlm_reference.COMMIT),
             **arrays)
    print(f"{FIXTURE}: {np.count_nonzero(counts)} of {counts.size} responses drawn")


if __name__ == "__main__":
    main()
