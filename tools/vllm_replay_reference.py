"""vLLM's replay of `tests/fixtures/hf/qwen3-moe-vllm`, the one implementation the `vllm-replay` site
records by (tools/vllm_replay_worker.py): vLLM serves the checkpoint in bf16 with top-k and top-p
filtering, `logprobs_mode="processed_logprobs"`, `return_sampling_mask` and
`enable_return_routed_experts` (`generated`), and transformers scores each sampled id on the routing vLLM
recorded, tempered and renormalized over the support vLLM kept (`scored`). Dew never imports vLLM.

Run in a vLLM environment, from the repository root, it writes the checkpoint from `CONFIG` with
transformers' own initialization under seed 0:

    python tools/vllm_replay_reference.py

The checkpoint is a Qwen3-MoE with a 64-wide attention head: vLLM's attention kernels on sm80/sm86 spin
forever on `qwen3-moe-tiny`'s 8-wide heads (flash_attn_varlen_func on an A100, Triton on a 3090).
"""

import functools
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "tests/fixtures/hf/qwen3-moe-vllm"
NEW_TOKENS = 10
SAMPLING = {"temperature": 0.8, "top_k": 6, "top_p": 0.9}
CONFIG = {"architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe", "vocab_size": 128,
          "hidden_size": 64, "intermediate_size": 48, "moe_intermediate_size": 16, "num_hidden_layers": 3,
          "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 64, "num_experts": 4,
          "num_experts_per_tok": 2, "decoder_sparse_step": 1, "mlp_only_layers": [0], "norm_topk_prob": False,
          "max_position_embeddings": 128, "rms_norm_eps": 1e-6, "rope_theta": 1e6,
          "tie_word_embeddings": False, "initializer_range": 0.2, "torch_dtype": "float32"}


def make_model() -> None:
    import torch
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    torch.manual_seed(0)
    Qwen3MoeForCausalLM(Qwen3MoeConfig(**CONFIG)).save_pretrained(MODEL, safe_serialization=True)


@functools.cache
def engine():
    """vLLM's engine over the checkpoint in bf16, returning what Dew replays."""
    from vllm import LLM

    return LLM(model=str(MODEL), skip_tokenizer_init=True, dtype="bfloat16", enforce_eager=True,
               max_model_len=128, gpu_memory_utilization=0.15, seed=0, enable_return_routed_experts=True,
               return_sampling_mask=True, logprobs_mode="processed_logprobs", enable_prefix_caching=False)


def generated(prompts: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """vLLM's completions of `prompts` `[rows, length]` under sampling `seed`, NEW_TOKENS ids a row, EOS
    ignored: `sampled_ids` `[rows, NEW_TOKENS]`, `behavior_log_probs` (vLLM's processed bf16 ones),
    `support` (each sampled id's kept ids, padded with -1 to top-k) and `routed_experts` `[rows, length +
    NEW_TOKENS - 1, layers, top_k]`, every decoder layer indexed, the dense layer 0 by the capture
    buffer's zeros."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    top_k = SAMPLING["top_k"]
    params = SamplingParams(max_tokens=NEW_TOKENS, logprobs=0, seed=seed, detokenize=False, ignore_eos=True,
                            **SAMPLING)
    requests = engine().generate([TokensPrompt(prompt_token_ids=row.tolist()) for row in prompts], params)
    sampled, behavior, support, routed = [], [], [], []
    for request in requests:
        output = request.outputs[0]
        ids = list(output.token_ids)
        sampled.append(ids)
        behavior.append([entry[token].logprob for entry, token in zip(output.logprobs, ids, strict=True)])
        kept = [list(map(int, row)) for row in output.sampling_mask.token_ids]
        support.append([row + [-1] * (top_k - len(row)) for row in kept])
        routed.append(np.asarray(output.routed_experts))
    shape = (prompts.shape[1] + NEW_TOKENS - 1, CONFIG["num_hidden_layers"], CONFIG["num_experts_per_tok"])
    if any(record.shape != shape for record in routed):
        raise ValueError(f"vLLM's routed_experts are {[record.shape for record in routed]}; the site reads "
                         f"{shape}")
    return {"sampled_ids": np.asarray(sampled, np.int32),
            "behavior_log_probs": np.asarray(behavior, np.float32),
            "support": np.asarray(support, np.int32), "routed_experts": np.asarray(routed, np.int32)}


def scored(model, prompts: np.ndarray, replay: dict[str, np.ndarray]) -> np.ndarray:
    """transformers' `model`'s log-probabilities of each sampled id of `replay` (`generated`'s) on the
    recorded routing (`_forced` on every sparse layer), tempered and renormalized over its kept support,
    in the model's dtype."""
    import torch

    sparse = [layer for layer in range(CONFIG["num_hidden_layers"]) if layer not in CONFIG["mlp_only_layers"]]
    rows = []
    for row, prompt in enumerate(prompts):
        ids = [*prompt.tolist(), *replay["sampled_ids"][row].tolist()]
        routed = torch.as_tensor(replay["routed_experts"][row], dtype=torch.long)
        for layer in sparse:
            gate = model.model.layers[layer].mlp.gate
            gate.forward = _forced(gate, routed[:, layer])
        with torch.no_grad():
            logits = model(torch.tensor([ids[:-1]])).logits[0] / SAMPLING["temperature"]
        start = len(prompt) - 1
        scores = []
        for offset, (token, kept) in enumerate(zip(replay["sampled_ids"][row], replay["support"][row],
                                                   strict=True)):
            kept = [int(index) for index in kept if index >= 0]
            line = logits[start + offset, kept]
            scores.append(line[kept.index(int(token))] - torch.logsumexp(line, 0))
        rows.append(torch.stack(scores))
    return torch.stack(rows).numpy()


def _forced(gate, indices):
    """`Qwen3MoeTopKRouter.forward` choosing `indices` instead of its top-k."""
    import torch
    import torch.nn.functional as F

    def forward(hidden_states):
        logits = F.linear(hidden_states.reshape(-1, gate.hidden_dim), gate.weight)
        probabilities = F.softmax(logits, dtype=torch.float, dim=-1)
        return logits, probabilities.gather(-1, indices).to(logits.dtype), indices

    return forward


if __name__ == "__main__":
    make_model()
