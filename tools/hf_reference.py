#!/usr/bin/env python3
"""Write the Hugging Face fixtures tests/test_hf_decoders.py checks against.

Everything here runs under torch and transformers, which dew does not depend
on, so this is the only place where the reference implementation is executed.
The fixtures it writes are what CI compares against.

Set up the venv and run it:

    uv venv /tmp/hfref --python 3.12
    uv pip install --python /tmp/hfref/bin/python torch \
        --index-url https://download.pytorch.org/whl/cpu
    uv pip install --python /tmp/hfref/bin/python transformers safetensors \
        sentencepiece numpy
    /tmp/hfref/bin/python tools/hf_reference.py

What lands in tests/fixtures/hf:

- qwen3-tiny/, gemma3-tiny/ and llama-tiny/: a random-weight checkpoint in the HF layout
  (config.json + model.safetensors), the 2 x 12 token ids it was run on, and
  the fp32 logits of the reference model in eval mode with eager attention.
  Small enough to live in git.
- qwen3-0.6b/: no weights. tensors.json is the tensor table of the real
  checkpoint straight from the hub metadata API, so a test can check the
  parameter tree without downloading 1.5 GB. prompt.json holds a 48 token
  prompt and reference.npz the top 32 logits per position of the real
  weights in fp32, which the network test compares against.
- gemma3-1b/: config.json only. google/gemma-3-1b-pt is gated and returns 401
  without a token, so the config comes from a mirror of it, which is the same
  file minus the mirror's own marker key.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import get_safetensors_metadata, hf_hub_download
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Gemma3ForCausalLM, Gemma3TextConfig,
    LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM,
)

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "hf"
REAL_MODEL = "Qwen/Qwen3-0.6B"
# google/gemma-3-1b-pt is gated; this mirror carries the identical config
GEMMA_MIRROR = "unsloth/gemma-3-1b-pt"
PROMPT = (
    "The Cascade Range runs from northern California through Oregon and "
    "Washington into British Columbia, and its volcanoes include Mount Rainier, "
    "Mount Hood and Mount St. Helens, which erupted in 1980. The tallest of "
    "them is the"
)
PROMPT_TOKENS = 48
TOP_K = 32
BATCH, LENGTH = 2, 12


def tiny_qwen3() -> Qwen3ForCausalLM:
    config = Qwen3Config(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, intermediate_size=128, vocab_size=256,
        tie_word_embeddings=True, rope_theta=1e6, max_position_embeddings=64,
        rms_norm_eps=1e-6, attention_bias=False, hidden_act="silu")
    torch.manual_seed(0)
    return Qwen3ForCausalLM(config)


def tiny_llama() -> LlamaForCausalLM:
    """Untied head and biased projections: the two switches Qwen3 leaves off.

    Llama applies config.attention_bias to all four projections, which is
    what CausalSelfAttention's one flag means, so a biased fixture is the
    test that the bias path loads.
    """
    config = LlamaConfig(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, intermediate_size=128, vocab_size=256,
        tie_word_embeddings=False, rope_theta=5e5, max_position_embeddings=64,
        rms_norm_eps=1e-5, attention_bias=True, mlp_bias=False, hidden_act="silu")
    torch.manual_seed(0)
    return LlamaForCausalLM(config)


def tiny_gemma3() -> Gemma3ForCausalLM:
    config = Gemma3TextConfig(
        hidden_size=64, num_hidden_layers=2,
        layer_types=["sliding_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=1, head_dim=32,
        query_pre_attn_scalar=16, sliding_window=4, intermediate_size=128,
        vocab_size=256, tie_word_embeddings=True, rope_theta=1e6,
        rope_local_base_freq=1e4, final_logit_softcapping=30.0,
        max_position_embeddings=64, rms_norm_eps=1e-6,
        hidden_activation="gelu_pytorch_tanh")
    torch.manual_seed(0)
    return Gemma3ForCausalLM(config)


def scatter_weights(model: torch.nn.Module) -> None:
    """Random weights with something in every tensor.

    A freshly constructed model leaves the RMSNorm scales at their identity
    value, and a fixture whose norms are all ones or all zeros would pass a
    parity test that had the (1 + w) offset backwards.
    """
    generator = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        for name, tensor in model.named_parameters():
            noise = torch.randn(tensor.shape, generator=generator) * 0.05
            tensor.copy_(tensor + noise if "norm" in name or "layernorm" in name
                         else noise * 4.0)


def reference_logits(model: torch.nn.Module, ids: np.ndarray) -> np.ndarray:
    model.eval()
    model.set_attn_implementation("eager")
    with torch.no_grad():
        out = model(input_ids=torch.from_numpy(ids), use_cache=False)
    return out.logits.to(torch.float32).numpy()


def write_tiny(name: str, model: torch.nn.Module) -> None:
    directory = FIXTURES / name
    directory.mkdir(parents=True, exist_ok=True)
    scatter_weights(model)
    model = model.to(torch.float32)
    model.save_pretrained(directory, safe_serialization=True)

    ids = np.random.RandomState(7).randint(
        0, model.config.vocab_size, (BATCH, LENGTH)).astype(np.int32)
    np.save(directory / "input_ids.npy", ids)
    np.save(directory / "logits.npy", reference_logits(model, ids))
    size = sum(path.stat().st_size for path in directory.iterdir())
    print(f"{directory}: {size / 1e3:.0f} kB, {sorted(p.name for p in directory.iterdir())}")


def write_tensor_table(directory: Path) -> None:
    """The real checkpoint's config, tensor names, shapes and dtypes.

    No weights: the hub serves this table without them, so a test can hold
    the parameter tree of a 1.5 GB checkpoint to account.
    """
    metadata = get_safetensors_metadata(REAL_MODEL)
    tensors = {}
    for file_metadata in metadata.files_metadata.values():
        for tensor_name, info in file_metadata.tensors.items():
            tensors[tensor_name] = {"shape": list(info.shape), "dtype": info.dtype}
    payload = {"repo": REAL_MODEL, "tensors": dict(sorted(tensors.items()))}
    (directory / "tensors.json").write_text(json.dumps(payload, indent=1) + "\n")

    config = hf_hub_download(REAL_MODEL, "config.json")
    (directory / "config.json").write_text(Path(config).read_text())
    print(f"{directory / 'tensors.json'}: {len(tensors)} tensors, with config.json")


def write_real_reference(directory: Path) -> None:
    """A prompt, and what the real weights predict for every position of it."""
    tokenizer = AutoTokenizer.from_pretrained(REAL_MODEL)
    ids = tokenizer(PROMPT, return_tensors="np")["input_ids"][:, :PROMPT_TOKENS]
    if ids.shape[1] != PROMPT_TOKENS:
        raise SystemExit(
            f"the prompt tokenizes to {ids.shape[1]} ids, not {PROMPT_TOKENS}")
    (directory / "prompt.json").write_text(json.dumps({
        "repo": REAL_MODEL, "prompt": PROMPT,
        "input_ids": ids[0].tolist(),
    }, indent=1) + "\n")

    model = AutoModelForCausalLM.from_pretrained(REAL_MODEL, dtype=torch.float32)
    logits = reference_logits(model, ids.astype(np.int64))[0]
    order = np.argsort(-logits, axis=-1)[:, :TOP_K]
    np.savez(
        directory / "reference.npz",
        top_ids=order.astype(np.int32),
        top_logits=np.take_along_axis(logits, order, axis=-1).astype(np.float32),
        argmax=np.argmax(logits, axis=-1).astype(np.int32))
    print(f"{directory / 'reference.npz'}: {logits.shape[0]} positions, "
          f"top {TOP_K}, argmax[:8]={np.argmax(logits, axis=-1)[:8].tolist()}")


def write_gemma3_config(directory: Path) -> None:
    """The real Gemma 3 1B text config, from a mirror of the gated repo.

    google/gemma-3-1b-pt answers 401 without an accepted licence, and the
    translation still has to be tested against a real Gemma config rather
    than only the tiny fixture, so this takes the mirror's copy and drops the
    marker key the mirror adds.
    """
    directory.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(hf_hub_download(GEMMA_MIRROR, "config.json")).read_text())
    config.pop("unsloth_fixed", None)
    (directory / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    print(f"{directory / 'config.json'}: {GEMMA_MIRROR}, "
          f"{config['num_hidden_layers']} layers, {len(config)} fields")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-real", action="store_true",
                        help="only the tiny fixtures, no 1.5 GB download")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    write_tiny("qwen3-tiny", tiny_qwen3())
    write_tiny("gemma3-tiny", tiny_gemma3())
    write_tiny("llama-tiny", tiny_llama())

    real = FIXTURES / "qwen3-0.6b"
    real.mkdir(parents=True, exist_ok=True)
    write_tensor_table(real)
    if not args.skip_real:
        write_real_reference(real)

    write_gemma3_config(FIXTURES / "gemma3-1b")


if __name__ == "__main__":
    main()
