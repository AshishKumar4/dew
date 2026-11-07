#!/usr/bin/env python3
"""Write the bfloat16 attention fixture and the transformers `generate` fixtures that
tests/test_generate_reference.py checks against.

Everything runs under torch and transformers 5.16.1 on CPU, against the
checkpoints already committed under tests/fixtures/hf, so no fixture weight
changes. DiffusionGemma's bfloat16 denoiser is tools/diffusion_gemma_worker.py's
recording over drawn inputs.

    PYTHONPATH=src:tools python tools/numerics_reference.py

What lands, beside each fixture's own files:

- tests/fixtures/attention_bf16.npz: one head of 512 queries over 512 keys
  of width 64, logits of standard deviation 0.5, through transformers'
  `eager_attention_forward` (the fp32 softmax every decoder's eager path
  runs) in bfloat16 and in float64. Inputs and the bf16 output are stored
  as bfloat16 bit patterns (uint16), the float64 output as float32, which
  is exact far below any bf16 difference.
- generate.npz in llama-tiny, qwen3-tiny, gemma3-tiny, mixtral-tiny and
  deepseek-v3-tiny: three left-padded prompts of different lengths, then
  `GenerationMixin.generate` from them in fp32 with a cache, once greedy
  and once sampled (temperature 0.7, top-k 20, top-p 0.9, torch seed 5).
  Per path: the tokens, the fp32 logits generate scored each step with
  (`output_logits`), the float64 logits of the same model teacher-forced
  over the same path at the same positions, and for the sampled path the
  selected token's log probability under the filtered distribution
  generate drew from, recomputed in float64 over the same filters. No EOS is configured, so every row
  runs the whole budget.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from diffusion_gemma_worker import eager
from torch_precision import float64
from transformers import AutoModelForCausalLM
from transformers.generation.logits_process import TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "hf"
GENERATORS = ("llama-tiny", "qwen3-tiny", "gemma3-tiny", "mixtral-tiny", "deepseek-v3-tiny")
PROMPT_LENGTHS = (4, 7, 5)
NEW_TOKENS = 10
TEMPERATURE, TOP_K, TOP_P, SAMPLE_SEED = 0.7, 20, 0.9, 5
PAD = 0


def decoder(name: str, dtype: torch.dtype):
    return eager(AutoModelForCausalLM.from_pretrained(FIXTURES / name, dtype=dtype,
                                                      local_files_only=True))


def logits(model, ids: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        out = model(input_ids=torch.from_numpy(ids).long(), use_cache=False).logits
    return out.to(torch.float64).numpy()


def prompts(vocab: int) -> tuple[np.ndarray, np.ndarray]:
    """Three prompts of different lengths, left-padded as transformers pads
    for generation, with ids clear of the low special tokens."""
    rng = np.random.RandomState(17)
    width = max(PROMPT_LENGTHS)
    ids = np.full((len(PROMPT_LENGTHS), width), PAD, np.int64)
    mask = np.zeros((len(PROMPT_LENGTHS), width), np.int64)
    for row, length in enumerate(PROMPT_LENGTHS):
        ids[row, width - length:] = rng.randint(3, vocab, length)
        mask[row, width - length:] = 1
    return ids, mask


def teacher_forced(model, sequences: np.ndarray, mask: np.ndarray, width: int) -> np.ndarray:
    """The logits scoring every generated slot, from one forward per row
    over its real tokens alone, so its first real token is position zero as
    in generate. Padding stays out of the forward: in float64 the eager
    mask sends a padded query's softmax to NaN, and a zero weight on its
    NaN value still poisons the real queries."""
    rows = []
    for row, valid in zip(sequences, mask, strict=True):
        real = row[width - int(valid.sum()):]
        with torch.no_grad():
            out = model(input_ids=torch.from_numpy(real)[None], use_cache=False).logits[0]
        rows.append(out[int(valid.sum()) - 1:-1].to(torch.float64).numpy())
    return np.stack(rows)


def filtered(scores: torch.Tensor) -> torch.Tensor:
    """generate's sampling chain in its own order: temperature, top-k, top-p."""
    for warper in (TemperatureLogitsWarper(TEMPERATURE), TopKLogitsWarper(TOP_K),
                   TopPLogitsWarper(TOP_P)):
        scores = warper(None, scores)
    return scores


def write_generation(name: str) -> None:
    directory = FIXTURES / name
    model = decoder(name, torch.float32)
    model.generation_config.eos_token_id = None
    ids, mask = prompts(model.config.vocab_size)
    width = ids.shape[1]
    common = {"input_ids": torch.from_numpy(ids), "attention_mask": torch.from_numpy(mask),
              "max_new_tokens": NEW_TOKENS, "min_new_tokens": NEW_TOKENS, "pad_token_id": PAD,
              "eos_token_id": None, "use_cache": True, "return_dict_in_generate": True,
              "output_logits": True}
    with torch.no_grad():
        greedy = model.generate(**common, do_sample=False, num_beams=1)
        torch.manual_seed(SAMPLE_SEED)
        sampled = model.generate(**common, do_sample=True, temperature=TEMPERATURE, top_k=TOP_K,
                                 top_p=TOP_P)
    arrays: dict[str, np.ndarray] = {"prompt": ids, "mask": mask}
    with float64():
        exact = decoder(name, torch.float64)
        f64s = {path: teacher_forced(exact, result.sequences.numpy(), mask, width)
                for path, result in (("greedy", greedy), ("sampled", sampled))}
    for path, result in (("greedy", greedy), ("sampled", sampled)):
        sequences = result.sequences.numpy()
        assert sequences.shape == (ids.shape[0], width + NEW_TOKENS)
        step_logits = torch.stack(result.logits, 1)
        f64 = f64s[path]
        arrays[f"{path}_tokens"] = sequences[:, width:]
        arrays[f"{path}_logits"] = step_logits.to(torch.float32).numpy()
        arrays[f"{path}_f64"] = f64
        report_generation(name, path, arrays[f"{path}_logits"], f64, arrays[f"{path}_tokens"])
    chosen = torch.from_numpy(arrays["sampled_tokens"])[..., None]
    exact_scores = filtered(torch.from_numpy(arrays["sampled_f64"]).reshape(-1, model.config.vocab_size))
    exact_behavior = torch.take_along_dim(
        exact_scores.reshape(arrays["sampled_f64"].shape).log_softmax(-1), chosen, -1)[..., 0]
    arrays["sampled_behavior_f64"] = exact_behavior.numpy()
    if not np.all(np.isfinite(arrays["sampled_behavior_f64"])):
        raise SystemExit(f"{name}: a sampled token lies outside the float64 filtered support")
    np.savez(directory / "generate.npz", **arrays)


def report_generation(name: str, path: str, fp32: np.ndarray, f64: np.ndarray,
                      tokens: np.ndarray) -> None:
    ranked = np.sort(f64, -1)
    margin = float(np.min(ranked[..., -1] - ranked[..., -2]))
    greedy = bool(np.array_equal(np.argmax(f64, -1), tokens))
    print(f"{name} {path}: fp32 decode error {np.max(np.abs(fp32 - f64)):.3e}, "
          f"smallest float64 top-2 margin {margin:.3e}, float64 argmax is the path {greedy}")
    if path == "greedy" and not greedy:
        raise SystemExit(f"{name}: the fp32 greedy path leaves the float64 argmax")


def bits(values: torch.Tensor) -> np.ndarray:
    return values.to(torch.bfloat16).contiguous().view(torch.int16).numpy().view(np.uint16)


def write_attention() -> None:
    from types import SimpleNamespace

    from transformers.models.llama.modeling_llama import eager_attention_forward

    rng = np.random.RandomState(0)
    width = 64
    # [batch, heads, length, width]; q.k / sqrt(width) has standard deviation 0.5
    q, k, v = (torch.from_numpy(rng.randn(1, 1, 512, width)).to(torch.bfloat16) for _ in range(3))
    k = (k.to(torch.float64) * 0.5).to(torch.bfloat16)
    module = SimpleNamespace(num_key_value_groups=1, training=False)

    def attend(dtype):
        out, _ = eager_attention_forward(module, q.to(dtype), k.to(dtype), v.to(dtype), None,
                                         scaling=width ** -0.5)
        return out
    with float64():
        exact = attend(torch.float64)
    np.savez(FIXTURES.parent / "attention_bf16.npz", q=bits(q), k=bits(k), v=bits(v),
             bf16=bits(attend(torch.bfloat16)), f64=exact.to(torch.float32).numpy())


def main() -> None:
    from reference_releases import require

    require("transformers")
    for name in GENERATORS:
        write_generation(name)
    write_attention()


if __name__ == "__main__":
    main()
