"""Record vLLM's routing and sampling-support fixture for Dew's replay tests.

Run in a vLLM environment on a GPU, from the repository root:

    python tools/vllm_replay_reference.py

It serves `tests/fixtures/hf/qwen3-moe-vllm` in bf16 with top-k and top-p
filtering, `logprobs_mode="processed_logprobs"`, `return_sampling_mask` and
`enable_return_routed_experts`, and writes what the engine returned, per
completion, to `tests/fixtures/rl/vllm_replay.json`: prompt and sampled ids,
the filtered behavior log-probabilities, each sampled id's kept support and
the `[forwarded ids, layers, top_k]` expert record. Dew never imports vLLM;
`tests/test_engine_replay.py` reads the file.

The checkpoint is written here, from `CONFIG` with transformers' own
initialization under seed 0, when it is missing. It is a Qwen3-MoE with a
64-wide attention head: vLLM's attention kernels on sm80/sm86 spin forever on
`qwen3-moe-tiny`'s 8-wide heads (flash_attn_varlen_func on an A100, Triton
on a 3090).
"""

import json
from importlib.metadata import version
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "tests/fixtures/hf/qwen3-moe-vllm"
OUT = ROOT / "tests/fixtures/rl/vllm_replay.json"
SAMPLING = {"temperature": 0.8, "top_k": 6, "top_p": 0.9}
CONFIG = {"architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe", "vocab_size": 128,
          "hidden_size": 64, "intermediate_size": 48, "moe_intermediate_size": 16, "num_hidden_layers": 3,
          "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 64, "num_experts": 4,
          "num_experts_per_tok": 2, "decoder_sparse_step": 1, "mlp_only_layers": [0], "norm_topk_prob": False,
          "max_position_embeddings": 128, "rms_norm_eps": 1e-6, "rope_theta": 1e6, "tie_word_embeddings": False,
          "initializer_range": 0.2, "torch_dtype": "float32"}


def make_model() -> None:
    import torch
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    torch.manual_seed(0)
    Qwen3MoeForCausalLM(Qwen3MoeConfig(**CONFIG)).save_pretrained(MODEL, safe_serialization=True)


def main() -> None:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    if not (MODEL / "model.safetensors").exists():
        make_model()
    engine = LLM(model=str(MODEL), skip_tokenizer_init=True, dtype="bfloat16", enforce_eager=True,
                 max_model_len=128, gpu_memory_utilization=0.15, seed=0,
                 enable_return_routed_experts=True, return_sampling_mask=True,
                 logprobs_mode="processed_logprobs", enable_prefix_caching=False)
    rng = np.random.default_rng(0)
    prompts = [TokensPrompt(prompt_token_ids=rng.integers(0, 128, size=size).tolist()) for size in (5, 9, 7, 12)]
    params = SamplingParams(max_tokens=10, logprobs=0, seed=3, detokenize=False, ignore_eos=True, **SAMPLING)
    calls = []
    for request in engine.generate(prompts, params):
        for output in request.outputs:
            sampled = list(output.token_ids)
            behavior = [entry[token].logprob for entry, token in zip(output.logprobs, sampled, strict=True)]
            support = [list(map(int, kept)) for kept in output.sampling_mask.token_ids]
            routed = np.asarray(output.routed_experts)
            calls.append({"prompt_ids": list(request.prompt_token_ids), "sampled_ids": sampled,
                          "behavior_log_probs": behavior, "support": support,
                          "routed_experts": routed.astype(int).tolist(), "routed_dtype": str(routed.dtype)})
    OUT.write_text(json.dumps({"vllm": version("vllm"), "model": MODEL.name, "sampling": SAMPLING,
                               "calls": calls}) + "\n")
    print(f"wrote {len(calls)} calls to {OUT}")


if __name__ == "__main__":
    main()
