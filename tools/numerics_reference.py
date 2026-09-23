#!/usr/bin/env python3
"""Write the float64 and bfloat16 reference fixtures and the transformers
`generate` fixtures that tests/test_bf16_reference.py,
tests/test_generate_reference.py and tests/test_block_diffusion.py check
against.

Everything runs under torch and transformers 5.16.1 on CPU, against the
checkpoints already committed under tests/fixtures/hf, so no fixture weight
changes. Each fixture's fp32 reference output is recomputed and placed
against the float64 one beside the committed output; a committed output
further from float64 than twice the fresh one stops the script, since the new
files would then describe different weights or a different reference.

    uv venv ~/.cache/dew/reference-venvs/numerics --python 3.12
    uv pip install --python ~/.cache/dew/reference-venvs/numerics/bin/python \
        torch --index-url https://download.pytorch.org/whl/cpu
    uv pip install --python ~/.cache/dew/reference-venvs/numerics/bin/python \
        transformers==5.16.1 safetensors numpy huggingface_hub sentencepiece
    PYTHONPATH=tools ~/.cache/dew/reference-venvs/numerics/bin/python \
        tools/numerics_reference.py

What lands, beside each fixture's own files:

- numerics.npz in llama-tiny, qwen3-tiny, gemma3-tiny, mixtral-tiny,
  deepseek-v3-tiny and mamba2-tiny: `f64` and `bf16`, the reference forward
  on the committed `input_ids.npy` with weights and activations in float64
  and in bfloat16 (stored as float32, which holds every bfloat16 exactly).
  The float64 run is the truth both precisions are measured from.
- numerics.npz in diffusion-gemma-workflow: the same pair for the three
  denoiser forwards in its reference.npz (`bare`, `conditioned`,
  `image_logits`), and in diffusion-gemma-denoise-tiny for `ref_bare` and
  `ref_conditioned`.
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
import transformers
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
from transformers.generation.logits_process import TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "hf"
DECODERS = ("llama-tiny", "qwen3-tiny", "gemma3-tiny", "mixtral-tiny", "deepseek-v3-tiny",
            "mamba2-tiny")
GENERATORS = ("llama-tiny", "qwen3-tiny", "gemma3-tiny", "mixtral-tiny", "deepseek-v3-tiny")
PROMPT_LENGTHS = (4, 7, 5)
NEW_TOKENS = 10
TEMPERATURE, TOP_K, TOP_P, SAMPLE_SEED = 0.7, 20, 0.9, 5
PAD = 0


def same(name: str, recomputed: np.ndarray, committed: np.ndarray, f64: np.ndarray) -> None:
    """The committed output is an evaluation of these weights at its own
    precision: another torch build sums in another order, so the two need
    not agree bit for bit, but the committed one may sit no further from
    float64 than twice the fresh one does."""
    fresh, stored = (float(np.max(np.abs(value - f64))) for value in (recomputed, committed))
    print(f"{name}: committed {stored:.3e} and recomputed {fresh:.3e} from float64, "
          f"{float(np.max(np.abs(recomputed - committed))):.3e} apart")
    if stored > 2 * fresh:
        raise SystemExit(f"{name}: the committed fixture is not this reference's output")


def eager(model):
    """Eager attention, as tools/hf_reference.py ran it, and the per-expert
    loop, the one routed kernel that runs in float64 on CPU (grouped_mm
    takes no double), so all three precisions run one order of operations."""
    model.set_attn_implementation("eager")
    if model._can_set_experts_implementation():
        model.set_experts_implementation("eager")
    return model.eval()


def decoder(name: str, dtype: torch.dtype):
    return eager(AutoModelForCausalLM.from_pretrained(FIXTURES / name, dtype=dtype,
                                                      local_files_only=True))


def logits(model, ids: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        out = model(input_ids=torch.from_numpy(ids).long(), use_cache=False).logits
    return out.to(torch.float64).numpy()


def write_decoder(name: str) -> None:
    directory = FIXTURES / name
    ids = np.load(directory / "input_ids.npy")
    f64 = logits(decoder(name, torch.float64), ids)
    same(name, logits(decoder(name, torch.float32), ids).astype(np.float32),
         np.load(directory / "logits.npy"), f64)
    bf16 = logits(decoder(name, torch.bfloat16), ids).astype(np.float32)
    np.savez(directory / "numerics.npz", f64=f64, bf16=bf16)
    report(name, np.load(directory / "logits.npy"), bf16, f64)


def report(name: str, fp32: np.ndarray, bf16: np.ndarray, f64: np.ndarray) -> None:
    print(f"{name}: fp32 reference error {np.max(np.abs(fp32 - f64)):.3e}, "
          f"bf16 reference error {np.max(np.abs(bf16 - f64)):.3e}")


def workflow(dtype: torch.dtype):
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaForBlockDiffusion
    return eager(DiffusionGemmaForBlockDiffusion.from_pretrained(
        FIXTURES / "diffusion-gemma-workflow", dtype=dtype, local_files_only=True))


def workflow_outputs(model, reference, dtype: torch.dtype) -> dict[str, np.ndarray]:
    """The three forwards tools/diffusion_gemma_reference.py stores, from its
    own stored inputs."""
    from transformers.models.gemma4.image_processing_pil_gemma4 import convert_image_to_patches

    prompt, canvas = (torch.from_numpy(reference[key]) for key in ("prompt", "canvas"))
    previous = torch.from_numpy(reference["previous"]).to(dtype)
    patches = torch.from_numpy(np.stack([convert_image_to_patches(image, 2)
                                         for image in reference["pixels"][:, 0]])).to(dtype)
    positions = torch.tensor([[[0, 0], [1, 0], [0, 1], [1, 1]]]).expand(2, -1, -1)
    image_prompt = torch.from_numpy(reference["image_prompt"])
    with torch.no_grad():
        outputs = {
            "bare": model(input_ids=prompt, decoder_input_ids=canvas).logits,
            "conditioned": model(input_ids=prompt, decoder_input_ids=canvas,
                                 self_conditioning_logits=previous).logits,
            "image_logits": model(input_ids=image_prompt, decoder_input_ids=canvas,
                                  pixel_values=patches, image_position_ids=positions).logits,
        }
    return {key: value.to(torch.float64).numpy() for key, value in outputs.items()}


def write_workflow() -> None:
    directory = FIXTURES / "diffusion-gemma-workflow"
    with np.load(directory / "reference.npz") as stored:
        reference = {key: stored[key] for key in stored.files}
    f64 = workflow_outputs(workflow(torch.float64), reference, torch.float64)
    fp32 = workflow_outputs(workflow(torch.float32), reference, torch.float32)
    for key, value in fp32.items():
        same(f"workflow {key}", value.astype(np.float32), reference[key], f64[key])
    bf16 = workflow_outputs(workflow(torch.bfloat16), reference, torch.bfloat16)
    arrays = {f"{key}_f64": value for key, value in f64.items()}
    arrays.update({f"{key}_bf16": value.astype(np.float32) for key, value in bf16.items()})
    np.savez(directory / "numerics.npz", **arrays)
    for key in f64:
        report(f"workflow {key}", reference[key], arrays[f"{key}_bf16"], f64[key])


def denoiser(dtype: torch.dtype):
    """The model tools/hf_reference.py `write_diffusion_denoiser_tiny` built,
    holding the text weights it saved. Its vision tower is left at its
    initialisation: none of the stored forwards has an image in it."""
    from hf_reference import DIFFUSION_DENOISER_TEXT
    from transformers.models.diffusion_gemma.configuration_diffusion_gemma import (
        DiffusionGemmaConfig,
        DiffusionGemmaTextConfig,
    )
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaForBlockDiffusion
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig

    text = DiffusionGemmaTextConfig(**{key: value for key, value in DIFFUSION_DENOISER_TEXT.items()
                                       if key != "model_type"})
    vision = Gemma4VisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, head_dim=16, patch_size=8,
        pooling_kernel_size=2, position_embedding_size=64)
    model = DiffusionGemmaForBlockDiffusion(DiffusionGemmaConfig(
        text_config=text, vision_config=vision, canvas_length=4, tie_word_embeddings=False))
    saved = load_file(FIXTURES / "diffusion-gemma-denoise-tiny" / "model.safetensors")
    missing, unexpected = model.load_state_dict(saved, strict=False)
    if unexpected or any(not name.startswith("model.encoder.vision_tower.")
                         and not name.startswith("model.encoder.embed_vision.") for name in missing):
        raise SystemExit(f"denoiser weights do not fit: missing {missing}, unexpected {unexpected}")
    return eager(model.to(dtype))


def denoiser_outputs(model, dtype: torch.dtype) -> dict[str, np.ndarray]:
    directory = FIXTURES / "diffusion-gemma-denoise-tiny"
    prompt = torch.from_numpy(np.load(directory / "prompt.npy")).long()
    canvas = torch.from_numpy(np.load(directory / "canvas.npy")).long()
    previous = torch.from_numpy(np.load(directory / "prev_logits.npy")).to(dtype)
    with torch.no_grad():
        bare = model(input_ids=prompt, decoder_input_ids=canvas).logits
        conditioned = model(input_ids=prompt, decoder_input_ids=canvas,
                            self_conditioning_logits=previous).logits
    return {"ref_bare": bare.to(torch.float64).numpy(),
            "ref_conditioned": conditioned.to(torch.float64).numpy()}


def write_denoiser() -> None:
    directory = FIXTURES / "diffusion-gemma-denoise-tiny"
    f64 = denoiser_outputs(denoiser(torch.float64), torch.float64)
    for key, value in denoiser_outputs(denoiser(torch.float32), torch.float32).items():
        same(f"denoiser {key}", value.astype(np.float32), np.load(directory / f"{key}.npy"), f64[key])
    bf16 = denoiser_outputs(denoiser(torch.bfloat16), torch.bfloat16)
    arrays = {f"{key}_f64": value for key, value in f64.items()}
    arrays.update({f"{key}_bf16": value.astype(np.float32) for key, value in bf16.items()})
    np.savez(directory / "numerics.npz", **arrays)
    for key in f64:
        report(f"denoiser {key}", np.load(directory / f"{key}.npy"), arrays[f"{key}_bf16"], f64[key])


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
    exact = decoder(name, torch.float64)
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
    for path, result in (("greedy", greedy), ("sampled", sampled)):
        sequences = result.sequences.numpy()
        assert sequences.shape == (ids.shape[0], width + NEW_TOKENS)
        step_logits = torch.stack(result.logits, 1)
        f64 = teacher_forced(exact, sequences, mask, width)
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
    np.savez(FIXTURES.parent / "attention_bf16.npz", q=bits(q), k=bits(k), v=bits(v),
             bf16=bits(attend(torch.bfloat16)), f64=attend(torch.float64).to(torch.float32).numpy())


def main() -> None:
    if transformers.__version__ != "5.16.1":
        raise SystemExit(f"the fixtures pin transformers 5.16.1, got {transformers.__version__}")
    for name in DECODERS:
        write_decoder(name)
    write_workflow()
    write_denoiser()
    for name in GENERATORS:
        write_generation(name)
    write_attention()


if __name__ == "__main__":
    main()
