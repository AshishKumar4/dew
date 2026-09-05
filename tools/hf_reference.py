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
- qwen3-tiny/, gemma3-tiny/, llama-tiny/, deepseek-v3-tiny/ and
  deepseek-v32-tiny/: a random-weight checkpoint in the HF layout
  (config.json + model.safetensors), the 2 x 12 token ids it was run on, and
  the fp32 logits of the reference model in eval mode with eager attention.
  Small enough to live in git. The DeepSeek pair is one dense layer over one
  MoE layer (`first_k_dense_replace` 1) with a shared expert, the released
  YaRN spelling, q and kv LoRA, and, on V3.2, the sparse indexer; their
  routers' balancing bias is scattered too, since a checkpoint carries it
  and a fixture at its zeros would not tell a load that reads it from one
  that drops it.
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
    AutoModelForCausalLM, AutoTokenizer, DeepseekV3Config, DeepseekV3ForCausalLM,
    Gemma3ForCausalLM, Gemma3TextConfig, LlamaConfig, LlamaForCausalLM,
    Qwen3Config, Qwen3ForCausalLM, MistralConfig, MistralForCausalLM, PreTrainedModel,
    MixtralConfig, MixtralForCausalLM, Qwen2Config, Qwen2ForCausalLM,
)
from transformers.models.deepseek_v32.configuration_deepseek_v32 import (
    DeepseekV32Config,
)
from transformers.models.deepseek_v32.modeling_deepseek_v32 import (
    DeepseekV32ForCausalLM,
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
    config = Qwen3Config.from_dict(dict(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, intermediate_size=128, vocab_size=256,
        tie_word_embeddings=True, rope_theta=1e6, max_position_embeddings=64,
        rms_norm_eps=1e-6, attention_bias=False, hidden_act="silu"))
    torch.manual_seed(0)
    return Qwen3ForCausalLM(config)


def tiny_llama() -> LlamaForCausalLM:
    """Untied head and biased projections: the two switches Qwen3 leaves off.

    Llama applies config.attention_bias to all four projections, which is
    what CausalSelfAttention's one flag means, so a biased fixture is the
    test that the bias path loads.
    """
    config = LlamaConfig.from_dict(dict(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, intermediate_size=128, vocab_size=256,
        tie_word_embeddings=False, rope_theta=5e5, max_position_embeddings=64,
        rms_norm_eps=1e-5, attention_bias=True, mlp_bias=False, hidden_act="silu"))
    torch.manual_seed(0)
    return LlamaForCausalLM(config)


def tiny_qwen2() -> Qwen2ForCausalLM:
    """Biased q/k/v projections with a bias-free o_proj, and a sliding window
    from the second layer on (use_sliding_window with max_window_layers)."""
    config = Qwen2Config.from_dict(dict(
        hidden_size=64, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=128, vocab_size=256,
        use_sliding_window=True, sliding_window=4, max_window_layers=1,
        max_position_embeddings=64, rope_theta=1e6, tie_word_embeddings=True))
    torch.manual_seed(0)
    return Qwen2ForCausalLM(config)


def tiny_mixtral() -> MixtralForCausalLM:
    config = MixtralConfig.from_dict(dict(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=48, vocab_size=128,
        num_local_experts=4, num_experts_per_tok=2, sliding_window=4,
        max_position_embeddings=64))
    torch.manual_seed(0)
    return MixtralForCausalLM(config)


def tiny_mistral() -> MistralForCausalLM:
    config = MistralConfig.from_dict(dict(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, intermediate_size=128, vocab_size=256,
        sliding_window=4, max_position_embeddings=64, rope_theta=10000.0))
    torch.manual_seed(0)
    return MistralForCausalLM(config)


def tiny_gemma3() -> Gemma3ForCausalLM:
    config = Gemma3TextConfig.from_dict(dict(
        hidden_size=64, num_hidden_layers=2,
        layer_types=["sliding_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=1, head_dim=32,
        query_pre_attn_scalar=16, sliding_window=4, intermediate_size=128,
        vocab_size=256, tie_word_embeddings=True, rope_theta=1e6,
        rope_local_base_freq=1e4, final_logit_softcapping=30.0,
        max_position_embeddings=64, rms_norm_eps=1e-6,
        hidden_activation="gelu_pytorch_tanh"))
    torch.manual_seed(0)
    return Gemma3ForCausalLM(config)


# The released rope spelling on both DeepSeek checkpoints: `rope_scaling`
# with `type` yarn, factor 40 off 4096 base positions, mscale on every dim.
DEEPSEEK_YARN = {
    "type": "yarn", "factor": 40.0, "beta_fast": 32, "beta_slow": 1,
    "mscale": 1.0, "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 4096,
}
# n_group 4 over 8 experts with topk_group 2 reaches four experts, which is
# the top_k, so the group limit decides the choice rather than its order.
DEEPSEEK_TINY = dict(
    vocab_size=256, hidden_size=32, intermediate_size=48,
    moe_intermediate_size=16, num_hidden_layers=2, num_attention_heads=4,
    num_key_value_heads=4, n_shared_experts=1, n_routed_experts=8,
    routed_scaling_factor=2.5, q_lora_rank=8, kv_lora_rank=8,
    qk_nope_head_dim=8, qk_rope_head_dim=8, v_head_dim=8, n_group=4,
    topk_group=2, num_experts_per_tok=4, first_k_dense_replace=1,
    norm_topk_prob=True, hidden_act="silu", max_position_embeddings=64,
    rms_norm_eps=1e-6, tie_word_embeddings=False, rope_theta=10000.0,
    rope_scaling=dict(DEEPSEEK_YARN), attention_bias=False)


def tiny_deepseek_v3() -> DeepseekV3ForCausalLM:
    config = DeepseekV3Config.from_dict(dict(DEEPSEEK_TINY, rope_interleave=True))
    torch.manual_seed(0)
    return DeepseekV3ForCausalLM(config)


# The v32 fixture's weights come from this seed rather than the family's
# 1234. The indexer scores a key at zero whenever every head's query-key
# agreement is negative (the relu), and torch.topk and jax.lax.top_k break
# an exact tie at the top-k boundary differently, so a fixture with a tie
# on any row compares two selections rather than two implementations. At
# two heads a quarter of the keys score zero and no seed in 3000 clears
# every row by more than 3.6e-3; at eight heads seed 202 keeps the fourth
# and fifth scores of every row of both layers at least 0.0217 apart.
DEEPSEEK_V32_SEED = 202


def tiny_deepseek_v32() -> DeepseekV32ForCausalLM:
    """The V3 shape with the sparse indexer: eight heads of width 16 over
    the rope width of 8, keeping four of the twelve keys."""
    config = DeepseekV32Config.from_dict(dict(
        DEEPSEEK_TINY, index_topk=4, index_n_heads=8, index_head_dim=16))
    torch.manual_seed(0)
    return DeepseekV32ForCausalLM(config)


def scatter_weights(model: torch.nn.Module, seed: int = 1234) -> None:
    """Random weights with something in every tensor.

    A freshly constructed model leaves the RMSNorm scales at their identity
    value, and a fixture whose norms are all ones or all zeros would pass a
    parity test that had the (1 + w) offset backwards.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, tensor in model.named_parameters():
            noise = torch.randn(tensor.shape, generator=generator) * 0.05
            tensor.copy_(tensor + noise if "norm" in name or "layernorm" in name
                         else noise * 4.0)
        # DeepSeek's balancing bias is a buffer the checkpoint carries, and
        # the reference selects on it: nonzero, or the load path that reads
        # it would agree with one that drops it.
        for name, tensor in model.named_buffers():
            if name.endswith("e_score_correction_bias"):
                tensor.copy_(torch.linspace(-0.4, 0.4, tensor.shape[0]))


def reference_logits(model: PreTrainedModel, ids: np.ndarray) -> np.ndarray:
    model.eval()
    model.set_attn_implementation("eager")
    with torch.no_grad():
        out = model(input_ids=torch.from_numpy(ids), use_cache=False)
    return out.logits.to(torch.float32).numpy()


def write_tiny(name: str, model: PreTrainedModel, seed: int = 1234) -> None:
    directory = FIXTURES / name
    directory.mkdir(parents=True, exist_ok=True)
    scatter_weights(model, seed)
    model = model.float()
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


def write_released_config(name: str, repo: str) -> None:
    """Download only the released config, never checkpoint weights."""
    directory = FIXTURES / name
    directory.mkdir(parents=True, exist_ok=True)
    source = Path(hf_hub_download(repo, "config.json"))
    (directory / "config.json").write_text(source.read_text())
    (directory / "source.json").write_text(json.dumps({"repo": repo}) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-real", action="store_true",
                        help="only the tiny fixtures, no 1.5 GB download")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    write_tiny("qwen3-tiny", tiny_qwen3())
    write_tiny("gemma3-tiny", tiny_gemma3())
    write_tiny("llama-tiny", tiny_llama())
    write_tiny("mistral-tiny", tiny_mistral())
    write_tiny("mixtral-tiny", tiny_mixtral())
    write_tiny("qwen2-tiny", tiny_qwen2())
    write_released_config("qwen2-0.5b", "Qwen/Qwen2-0.5B")
    write_released_config("mixtral-8x7b", "mistralai/Mixtral-8x7B-v0.1")
    write_released_config("mistral-7b-v0.3", "mistralai/Mistral-7B-v0.3")
    write_tiny("deepseek-v3-tiny", tiny_deepseek_v3())
    write_tiny("deepseek-v32-tiny", tiny_deepseek_v32(), seed=DEEPSEEK_V32_SEED)

    real = FIXTURES / "qwen3-0.6b"
    real.mkdir(parents=True, exist_ok=True)
    write_tensor_table(real)
    if not args.skip_real:
        write_real_reference(real)

    write_gemma3_config(FIXTURES / "gemma3-1b")


if __name__ == "__main__":
    main()
