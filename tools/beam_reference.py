#!/usr/bin/env python3
"""Write tests/fixtures/hf/<name>/beam.npz: transformers' beam search.

`GenerationMixin.generate` with `num_beams` (transformers 5.16.1's
`_beam_search`, fp32, eager attention, with its cache) on a committed tiny
checkpoint, from tools/numerics_reference.py's three left-padded prompts,
width 3, three returned beams per prompt, six new tokens. The EOS id is the
third token of the second beam a search without EOS returns for the first
prompt, so EOS ends beams inside the budget while others run to it. Four
settings of `length_penalty` and `early_stopping` cover the three
early-stopping rules and a penalty that rewards, ignores and punishes
length. A fifth search is shaped: a sequence bias on one token and on a
bigram the free search walks, then `renormalize_logits`, which transformers
appends after every other processor, at penalty 2.63 with early stopping
"never"; the tool checks that both the bias and the renormalization move a
beam.

Each prompt searches alone, unpadded: in float64 the eager mask sends a
padded query's softmax to NaN. Each search runs again on the float64 model
with `_beam_search`'s own float32 casts and float32 beam scores read as
float64 (`torch.float32` and `torch.float` in transformers' generation
module), and its beams must equal the float32 run's, so they are the
model's and not a rounding of it. What lands per setting: the returned
sequences (pad after EOS), their scores in float32 and float64
(`sequences_scores`, the summed log probability over the generated length
raised to the penalty), and the settings.

    PYTHONPATH=tools ~/.cache/dew/reference-venvs/numerics/bin/python \
        tools/beam_reference.py
"""

from __future__ import annotations

import json
import types

import numpy as np
import torch
import transformers
import transformers.generation.utils as generation
from numerics_reference import FIXTURES, decoder, prompts

FAMILIES = ("llama-tiny", "qwen3-tiny")
WIDTH, NEW_TOKENS = 3, 6
SETTINGS = ((1.0, False), (0.0, True), (2.0, "never"), (-1.0, False))
"""(length_penalty, early_stopping) pairs."""
SHAPED = (2.6265404784, "never")
BIASES = (2.5, 5.0)
"""Large enough that renormalizing moves a beam: the bias lifts the
normalizer of the rows it touches, which the renormalized scores pay."""


def searched(model, ids, mask, eos, pad, penalty, early, **shaping) -> tuple[np.ndarray, np.ndarray]:
    """Every prompt's `WIDTH` returned beams' generated tokens, padded with
    `pad` to the budget, and their scores, one prompt at a time."""
    width, tokens, scores = ids.shape[1], [], []
    wide = next(model.parameters()).dtype == torch.float64
    kept = generation.torch
    if wide:
        generation.torch = types.SimpleNamespace(**{**vars(torch), "float32": torch.float64,
                                                    "float": torch.float64})
    try:
        for row, valid in zip(ids, mask, strict=True):
            real = torch.from_numpy(row[width - int(valid.sum()):])[None]
            with torch.no_grad():
                result = model.generate(
                    input_ids=real, num_beams=WIDTH, num_return_sequences=WIDTH, max_new_tokens=NEW_TOKENS,
                    do_sample=False, length_penalty=penalty, early_stopping=early, eos_token_id=eos,
                    pad_token_id=pad, use_cache=True, return_dict_in_generate=True, output_scores=True,
                    **shaping)
            drawn = result.sequences.numpy()[:, real.shape[1]:]
            tokens.append(np.pad(drawn, ((0, 0), (0, NEW_TOKENS - drawn.shape[1])), constant_values=pad))
            scores.append(result.sequences_scores.to(torch.float64).numpy())
    finally:
        generation.torch = kept
    return np.concatenate(tokens), np.concatenate(scores)


def write(name: str) -> None:
    model, exact = decoder(name, torch.float32), decoder(name, torch.float64)
    for each in (model, exact):
        each.generation_config.eos_token_id = None
    ids, mask = prompts(model.config.vocab_size)
    free, _ = searched(model, ids, mask, None, 0, 1.0, early=False)
    eos = int(free[1, 2])
    pad = next(token for token in range(3, model.config.vocab_size)
               if token != eos and not np.isin(free, token).any())
    arrays: dict[str, np.ndarray] = {"eos": np.asarray(eos), "pad": np.asarray(pad),
                                     "settings": np.asarray(json.dumps(SETTINGS))}
    ended = 0
    for index, (penalty, early) in enumerate(SETTINGS):
        tokens, scores = searched(model, ids, mask, eos, pad, penalty, early)
        wide, scores_f64 = searched(exact, ids, mask, eos, pad, penalty, early)
        if not np.array_equal(tokens, wide):
            raise SystemExit(f"{name} {penalty} {early}: float32 and float64 searches return other beams")
        ended += int(np.isin(tokens, eos).any(-1).sum())
        arrays[f"case_{index}_tokens"] = tokens
        arrays[f"case_{index}_scores"] = scores
        arrays[f"case_{index}_scores_f64"] = scores_f64
    # The token the free search's best first beam draws third, and the
    # bigram its second beam draws at its fourth and fifth steps.
    bias = [[[int(free[0, 2])], BIASES[0]], [[int(free[1, 3]), int(free[1, 4])], BIASES[1]]]
    shaping = {"sequence_bias": bias, "renormalize_logits": True}
    tokens, _ = searched(model, ids, mask, eos, pad, *SHAPED, **shaping)
    wide, _ = searched(exact, ids, mask, eos, pad, *SHAPED, **shaping)
    if not np.array_equal(tokens, wide):
        raise SystemExit(f"{name} shaped: float32 and float64 searches return other beams")
    plain, _ = searched(model, ids, mask, eos, pad, *SHAPED)
    raw, _ = searched(model, ids, mask, eos, pad, *SHAPED, sequence_bias=bias, renormalize_logits=False)
    if np.array_equal(tokens, plain) or np.array_equal(tokens, raw):
        raise SystemExit(f"{name} shaped: the bias or the renormalization moves no beam")
    arrays["shaped_tokens"] = tokens
    arrays["shaped_bias"] = np.asarray(json.dumps(bias))
    arrays["shaped_setting"] = np.asarray(json.dumps(SHAPED))
    print(f"{name}: EOS {eos}, pad {pad}, {ended} of {len(SETTINGS) * len(ids) * WIDTH} beams end on EOS")
    if not ended:
        raise SystemExit(f"{name}: no beam ends on EOS")
    np.savez(FIXTURES / name / "beam.npz", **arrays)


def main() -> None:
    if transformers.__version__ != "5.16.1":
        raise SystemExit(f"the fixtures pin transformers 5.16.1, got {transformers.__version__}")
    for name in FAMILIES:
        write(name)


if __name__ == "__main__":
    main()
