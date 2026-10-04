#!/usr/bin/env python3
"""Write tests/fixtures/decoding/speculative.npz: transformers' speculative
acceptance on fixed distributions and fixed uniforms.

`_speculative_sampling` (transformers 5.16.1, generation/utils.py,
algorithm 1 of arXiv 2211.17192) verifies three drafted candidates against
the target's four distributions (one per candidate and the bonus) for 256
rows of an 8-token vocabulary. Its `torch.rand_like` returns, per row, the
uniforms Dew's `Speculative` draws for those candidates from
`jax.random.key(SEED)` split as the verification splits a row's keys, so the
two accept the same candidates; its `torch.multinomial` records the
distribution the next token is drawn from (the normalized positive part of
p - q after a rejection, p after a block accepted whole) instead of
drawing. Half the rows draft from a distribution near the target's, half
from an unrelated one, so blocks end at every length. What lands: the
logits and candidates, the matches, and the next token's distribution in
float32 and float64.

    python tools/speculative_reference.py
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import torch
import transformers
from transformers.generation.utils import _speculative_sampling

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "decoding" / "speculative.npz"
ROWS, BLOCK, VOCAB, SEED = 256, 4, 8, 11


def uniforms() -> np.ndarray:
    """The uniforms `Speculative` compares each candidate's ratio with: a
    row's keys split `2 * BLOCK + 1` ways, candidate `at`'s at `BLOCK + at`."""
    keys = jax.random.split(jax.random.key(SEED), ROWS * (2 * BLOCK + 1)).reshape(ROWS, 2 * BLOCK + 1)
    return np.stack([np.asarray(jax.vmap(jax.random.uniform)(keys[:, BLOCK + at]))
                     for at in range(1, BLOCK)], axis=1)


def verified(target, draft, candidates, drawn, dtype) -> tuple[int, np.ndarray]:
    """`_speculative_sampling` on one row, its uniforms `drawn`."""
    recorded = []
    kept = torch.rand_like, torch.multinomial
    torch.rand_like = lambda like: torch.as_tensor(drawn, dtype=like.dtype)

    def multinomial(probs, num_samples=1, **_):
        recorded.append(probs.detach().clone())
        return torch.zeros((probs.shape[0], num_samples), dtype=torch.long)

    torch.multinomial = multinomial
    try:
        ids = torch.as_tensor(np.concatenate([[0], candidates]), dtype=torch.long)[None]
        _, matches = _speculative_sampling(ids, torch.as_tensor(draft, dtype=dtype)[None], BLOCK - 1,
                                           torch.as_tensor(target, dtype=dtype)[None],
                                           is_done_candidate=False)
    finally:
        torch.rand_like, torch.multinomial = kept
    (probs,) = recorded
    return int(matches), probs[0].to(torch.float64).numpy()


def main() -> None:
    if transformers.__version__ != "5.16.1":
        raise SystemExit(f"the fixture pins transformers 5.16.1, got {transformers.__version__}")
    generator = np.random.default_rng(7)
    target = (generator.standard_normal((ROWS, BLOCK + 1, VOCAB)) * 1.5).astype(np.float32)
    unrelated = generator.standard_normal((ROWS, BLOCK, VOCAB)) * 1.5
    near = target[:, :BLOCK] + generator.standard_normal((ROWS, BLOCK, VOCAB)) * 0.3
    draft = np.where((np.arange(ROWS) % 2 == 0)[:, None, None], near, unrelated).astype(np.float32)
    probs = np.exp(draft - draft.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    candidates = np.stack([[generator.choice(VOCAB, p=probs[row, at]) for at in range(1, BLOCK)]
                           for row in range(ROWS)])
    drawn = uniforms()
    arrays = {"target": target, "draft": draft, "candidates": candidates, "uniforms": drawn,
              "seed": np.asarray(SEED)}
    for dtype, suffix in ((torch.float32, ""), (torch.float64, "_f64")):
        runs = [verified(target[row, 1:], draft[row, 1:], candidates[row], drawn[row], dtype)
                for row in range(ROWS)]
        arrays[f"matches{suffix}"] = np.asarray([matches for matches, _ in runs])
        arrays[f"next{suffix}"] = np.stack([distribution for _, distribution in runs])
    if not np.array_equal(arrays["matches"], arrays["matches_f64"]):
        raise SystemExit("float32 and float64 accept different candidates")
    print(f"{FIXTURE}: blocks keep {np.bincount(arrays['matches'], minlength=BLOCK).tolist()} candidates")
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(FIXTURE, **arrays)


if __name__ == "__main__":
    main()
