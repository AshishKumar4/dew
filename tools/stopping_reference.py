#!/usr/bin/env python3
"""Write tests/fixtures/hf/<name>/stopping.npz: transformers `generate`
stopping on EOS, on EOS held off by `min_new_tokens`, and on `max_length`.

Each run is `GenerationMixin.generate` (transformers 5.16.1, fp32, eager
attention, with its cache, greedy) on a committed tiny checkpoint from the
three left-padded prompts tools/numerics_reference.py builds:

- `eos`: two EOS ids, picked from the committed greedy path (generate.npz)
  so that row 0 draws one at its third step, row 1 at its fifth and row 2
  never, and a pad id apart from both: rows end at different steps and
  transformers pads each finished row's slots with the pad id.
- `min_new`: the same with `min_new_tokens` 4, which keeps row 0 from
  ending at its third step: its EOS logit is set to minus infinity until
  then, so the greedy path turns to its runner-up.
- `max_length`: each row alone, unpadded, to `max_length` 9 with no EOS,
  so a row of n prompt tokens draws 9 - n. `max_length` counts the input's
  width, so a left-padded batch would stop every row at the padded width's
  count; one row at a time is the length of the row's own tokens, which is
  what Dew's `MaxLength` counts.

Every path's float64 top-2 margin, over the scores transformers chose
from, is checked against fp32 rounding as tools/numerics_reference.py
checks its greedy path, so the path is the model's and not a rounding.

    PYTHONPATH=tools ~/.cache/dew/reference-venvs/numerics/bin/python \
        tools/stopping_reference.py
"""

from __future__ import annotations

import numpy as np
import torch
import transformers
from numerics_reference import FIXTURES, NEW_TOKENS, decoder, prompts, teacher_forced

FAMILIES = ("llama-tiny", "qwen3-tiny")
STOPS = (2, 4)
"""The step at which rows 0 and 1 draw an EOS id on the plain greedy path."""
MIN_NEW, MAX_LENGTH = 4, 9


def chosen(greedy: np.ndarray) -> tuple[int, int]:
    """Row 0's token at its third step and row 1's at its fifth, provided
    neither appears earlier on rows 0 and 1 or anywhere on row 2."""
    eos = (int(greedy[0, STOPS[0]]), int(greedy[1, STOPS[1]]))
    for row, stop in enumerate(STOPS):
        if np.isin(greedy[row, :stop], eos).any():
            raise SystemExit(f"row {row} draws an EOS id before step {stop}")
    if np.isin(greedy[2], eos).any():
        raise SystemExit("row 2 draws an EOS id")
    return eos


def margin(f64: np.ndarray, tokens: np.ndarray, banned: np.ndarray, label: str) -> float:
    """The smallest float64 top-2 margin over the scores the path chose
    from, with the ids `banned[step]` holds out of reach."""
    scores = np.where(banned, -np.inf, f64)
    if not np.array_equal(np.argmax(scores, -1), tokens):
        raise SystemExit(f"{label}: the fp32 path leaves the float64 argmax")
    ranked = np.sort(scores, -1)
    return float(np.min(ranked[..., -1] - ranked[..., -2]))


def write(name: str) -> None:
    directory = FIXTURES / name
    model = decoder(name, torch.float32)
    exact = decoder(name, torch.float64)
    model.generation_config.eos_token_id = None
    with np.load(directory / "generate.npz") as stored:
        greedy = stored["greedy_tokens"]
        # The fp32 decode's error on the committed greedy path, the rounding
        # every path here is measured against.
        error = float(np.max(np.abs(stored["greedy_logits"] - stored["greedy_f64"])))
    ids, mask = prompts(model.config.vocab_size)
    width, vocab = ids.shape[1], model.config.vocab_size
    eos = chosen(greedy)
    pad = next(token for token in range(3, vocab) if token not in eos and not np.isin(greedy, token).any())
    arrays: dict[str, np.ndarray] = {"eos": np.asarray(eos), "pad": np.asarray(pad),
                                     "min_new": np.asarray(MIN_NEW), "max_length": np.asarray(MAX_LENGTH)}
    common = {"input_ids": torch.from_numpy(ids), "attention_mask": torch.from_numpy(mask),
              "max_new_tokens": NEW_TOKENS, "pad_token_id": pad, "eos_token_id": list(eos),
              "do_sample": False, "num_beams": 1, "use_cache": True}
    margins = []
    for path, extra in (("eos", {}), ("min_new", {"min_new_tokens": MIN_NEW})):
        with torch.no_grad():
            sequences = model.generate(**common, **extra).numpy()
        tokens = sequences[:, width:]
        assert tokens.shape == (len(ids), NEW_TOKENS), tokens.shape
        f64 = teacher_forced(exact, sequences, mask, width)
        for row in range(len(ids)):
            ended = np.flatnonzero(np.isin(tokens[row], eos))
            length = int(ended[0]) + 1 if ended.size else NEW_TOKENS
            assert np.all(tokens[row, length:] == pad), (path, row)
            banned = np.zeros((length, vocab), bool)
            if path == "min_new":
                banned[:MIN_NEW, list(eos)] = True
            label = f"{name} {path} row {row}"
            margins.append(margin(f64[row, :length], tokens[row, :length], banned, label))
        arrays[f"{path}_tokens"] = tokens
    for row in range(len(ids)):
        real = ids[row, width - int(mask[row].sum()):]
        with torch.no_grad():
            sequence = model.generate(input_ids=torch.from_numpy(real)[None], max_length=MAX_LENGTH,
                                      pad_token_id=pad, eos_token_id=None, do_sample=False, num_beams=1,
                                      use_cache=True).numpy()[0]
        drawn = sequence[len(real):]
        assert len(drawn) == MAX_LENGTH - len(real), (row, len(drawn))
        f64 = teacher_forced(exact, sequence[None], np.ones((1, len(real)), np.int64), len(real))[0]
        margins.append(margin(f64, drawn, np.zeros(f64.shape, bool), f"{name} max_length row {row}"))
        arrays[f"max_length_tokens_{row}"] = drawn
    arrays["margin"] = np.asarray(min(margins))
    print(f"{name}: EOS {eos}, pad {pad}, smallest float64 top-2 margin {min(margins):.3e}, "
          f"fp32 decode error on the greedy path {error:.3e}")
    if min(margins) <= 4 * error:
        raise SystemExit(f"{name}: a margin within four fp32 decode errors")
    np.savez(directory / "stopping.npz", **arrays)


def main() -> None:
    if transformers.__version__ != "5.16.1":
        raise SystemExit(f"the fixtures pin transformers 5.16.1, got {transformers.__version__}")
    for name in FAMILIES:
        write(name)


if __name__ == "__main__":
    main()
