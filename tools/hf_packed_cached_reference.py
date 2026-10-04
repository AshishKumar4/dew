#!/usr/bin/env python3
"""Write the packed and cached decoding fixture with transformers' own models.

Over the committed llama-tiny (grouped heads, projection biases) and
mistral-tiny (a sliding window of 4) checkpoints, transformers 5.16.1's own
`LlamaForCausalLM` and `MistralForCausalLM`, eager attention, record:

- packed rows: several documents to a row, each restarting `position_ids`
  at zero and no attention mask, from which transformers builds its own
  packed mask (`masking_utils.find_packed_sequence_indices`); the tool
  checks that each document's logits are the ones it gets alone;
- cached decoding: a prompt prefilled into transformers' cache, then one
  token a step, past the sliding window, the logits of the prompt's last
  position and of every step; the tool checks they are the full sequence's.

Each in float32 and in float64 (the model built and run under
`diffusers_wan_reference.widened`, which widens its float32 pins, the
rotary table among them, and with the eager attention's
`softmax(dtype=torch.float32)` widened too), the truth
tests/reference_error.py measures both from. The packed call asks for no
cache: transformers looks for packing only when it builds none.

Run with the Dew test environment's torch and transformers, on CPU:

    python tools/hf_packed_cached_reference.py OUTPUT.npz
"""

import contextlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("llama-tiny", "mistral-tiny")
# Documents per packed row, each row 16 tokens.
PACKING = ((5, 7, 4), (9, 7))
PROMPT, STEPS = 5, 9
SEED = 83


def model(family: str, dtype: torch.dtype):
    return AutoModelForCausalLM.from_pretrained(ROOT / "tests" / "fixtures" / "hf" / family, dtype=dtype,
                                                attn_implementation="eager").eval()


@contextlib.contextmanager
def wide_softmax():
    """The eager attention's `softmax(..., dtype=torch.float32)` in float64."""
    softmax = torch.nn.functional.softmax

    def widened(*args, dtype=None, **kwargs):
        return softmax(*args, dtype=torch.float64 if dtype == torch.float32 else dtype, **kwargs)

    torch.nn.functional.softmax = widened
    try:
        yield
    finally:
        torch.nn.functional.softmax = softmax


def float64():
    from diffusers_wan_reference import widened

    stack = contextlib.ExitStack()
    stack.enter_context(widened())
    stack.enter_context(wide_softmax())
    return stack


def packed(net, ids: np.ndarray) -> torch.Tensor:
    positions = torch.tensor([[step for length in row for step in range(length)] for row in PACKING])
    return net(input_ids=torch.from_numpy(ids), position_ids=positions, use_cache=False).logits


def alone(net, ids: np.ndarray) -> torch.Tensor:
    """Each document run by itself, laid back out as the packed rows."""
    rows = []
    for row, lengths in zip(ids, PACKING, strict=True):
        starts = np.cumsum((0, *lengths[:-1]))
        rows.append(torch.cat([net(input_ids=torch.from_numpy(row[start:start + length][None])).logits[0]
                               for start, length in zip(starts, lengths, strict=True)]))
    return torch.stack(rows)


def cached(net, ids: np.ndarray) -> torch.Tensor:
    """The prompt's last logits and each step's, through transformers' cache."""
    output = net(input_ids=torch.from_numpy(ids[:, :PROMPT]), use_cache=True)
    cache, steps = output.past_key_values, [output.logits[:, -1]]
    for position in range(PROMPT, PROMPT + STEPS):
        output = net(input_ids=torch.from_numpy(ids[:, position:position + 1]), past_key_values=cache,
                     use_cache=True)
        cache = output.past_key_values
        steps.append(output.logits[:, -1])
    return torch.stack(steps, dim=1)


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    torch.set_num_threads(1)
    rng = np.random.default_rng(SEED)
    arrays: dict[str, np.ndarray] = {}
    for family in FAMILIES:
        config = ROOT / "tests" / "fixtures" / "hf" / family / "config.json"
        vocab = json.loads(config.read_text())["vocab_size"]
        packed_ids = rng.integers(0, vocab, (len(PACKING), 16)).astype(np.int64)
        decode_ids = rng.integers(0, vocab, (2, PROMPT + STEPS)).astype(np.int64)
        arrays.update({f"{family}/packed_ids": packed_ids, f"{family}/decode_ids": decode_ids})
        for precision, dtype, scope in (("fp32", torch.float32, contextlib.nullcontext),
                                        ("fp64", torch.float64, float64)):
            with scope(), torch.no_grad():
                net = model(family, dtype)
                logits = packed(net, packed_ids)
                gap = float((logits - alone(net, packed_ids)).abs().max())
                steps = cached(net, decode_ids)
                full = net(input_ids=torch.from_numpy(decode_ids)).logits[:, PROMPT - 1:]
                drift = float((steps - full).abs().max())
            assert logits.dtype == steps.dtype == dtype, (family, precision)
            if precision == "fp64":
                assert gap < 1e-12 and drift < 1e-12, (family, gap, drift)
            arrays[f"{family}/{precision}.packed"] = logits.numpy()
            arrays[f"{family}/{precision}.cached"] = steps.numpy()
            print(f"{family} {precision}: packed vs alone {gap:.2g}, cached vs full {drift:.2g}")
    meta = {"transformers": transformers.__version__, "torch": torch.__version__, "families": FAMILIES,
            "packing": PACKING, "prompt": PROMPT, "steps": STEPS}
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    np.savez_compressed(sys.argv[1], **arrays)
    print(f"{sys.argv[1]}: {Path(sys.argv[1]).stat().st_size / 1e6:.2f} MB, {len(arrays)} arrays")


if __name__ == "__main__":
    main()
