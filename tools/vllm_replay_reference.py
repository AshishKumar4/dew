"""Record vLLM's routing and sampling-support fixture for Dew's replay tests.

Run in a vLLM environment on a GPU, from the repository root:

    python tools/vllm_replay_reference.py
    python tools/vllm_replay_reference.py --reference   # only the transformers references, on the record

It serves `tests/fixtures/hf/qwen3-moe-vllm` in bf16 with top-k and top-p
filtering, `logprobs_mode="processed_logprobs"`, `return_sampling_mask` and
`enable_return_routed_experts`, and writes what the engine returned, per
completion, to `tests/fixtures/rl/vllm_replay.json`: prompt and sampled ids,
the filtered behavior log-probabilities, each sampled id's kept support and
the `[forwarded ids, layers, top_k]` expert record. Dew never imports vLLM;
`tests/test_engine_replay.py` reads the file.

Beside each call it writes `reference`: the recorded ids' filtered
log-probabilities from transformers at float32 and at float64 (the
`tests/reference_error.py` rule's reference and truth), each the capped
logit over the temperature renormalized over the recorded support. It
refuses a record whose routing transformers' own float64 routers would not
choose, so the references score what the engine routed.

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


def references(record: dict) -> None:
    """Add each call's transformers filtered log-probabilities: float32 and
    float64 on transformers' own routing, which must be the record's, and
    bfloat16 with the record's routing forced, so it differs from vLLM's
    bf16 run by rounding alone."""
    import torch
    from transformers import Qwen3MoeForCausalLM

    # One thread sums in one order, so the float32 references regenerate bit for bit.
    torch.set_num_threads(1)
    temperature = record["sampling"]["temperature"]
    sparse = [layer for layer in range(CONFIG["num_hidden_layers"]) if layer not in CONFIG["mlp_only_layers"]]
    for dtype, name in ((torch.float32, "float32"), (torch.float64, "float64"), (torch.bfloat16, "bfloat16")):
        # The grouped-matmul experts refuse float64; the eager loop computes the same sum.
        model = Qwen3MoeForCausalLM.from_pretrained(MODEL, dtype=dtype, experts_implementation="eager").eval()
        for call in record["calls"]:
            ids = call["prompt_ids"] + call["sampled_ids"]
            routed = torch.as_tensor(np.asarray(call["routed_experts"]), dtype=torch.long)
            if name == "bfloat16":
                for layer in sparse:
                    model.model.layers[layer].mlp.gate.forward = _forced(model.model.layers[layer].mlp.gate,
                                                                         routed[:, layer])
            with torch.no_grad():
                out = model(torch.tensor([ids[:-1]]), output_router_logits=True)
            if name != "bfloat16":
                for layer, logits in zip(sparse, out.router_logits, strict=True):
                    chosen = torch.topk(logits.softmax(-1), CONFIG["num_experts_per_tok"], dim=-1).indices
                    if not torch.equal(chosen.sort(-1).values, routed[:, layer].sort(-1).values):
                        raise ValueError(f"transformers {name} routes layer {layer} otherwise than the record")
            logits = out.logits[0].double() / temperature
            start = len(call["prompt_ids"]) - 1
            scores = []
            for offset, (token, kept) in enumerate(zip(call["sampled_ids"], call["support"], strict=True)):
                row = logits[start + offset, kept]
                scores.append(float(row[kept.index(token)] - torch.logsumexp(row, 0)))
            call.setdefault("reference", {})[name] = scores


def _forced(gate, indices):
    """`Qwen3MoeTopKRouter.forward` choosing `indices` instead of its top-k."""
    import torch
    import torch.nn.functional as F

    def forward(hidden_states):
        logits = F.linear(hidden_states.reshape(-1, gate.hidden_dim), gate.weight)
        probabilities = F.softmax(logits, dtype=torch.float, dim=-1)
        return logits, probabilities.gather(-1, indices).to(logits.dtype), indices

    return forward


def main() -> None:
    import sys

    if sys.argv[1:] == ["--reference"]:
        record = json.loads(OUT.read_text())
        references(record)
        OUT.write_text(json.dumps(record) + "\n")
        return
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
            # Every decoder layer is indexed, the dense layer 0 by the capture buffer's zeros.
            forwarded = len(request.prompt_token_ids) + len(sampled) - 1
            if routed.shape != (forwarded, CONFIG["num_hidden_layers"], CONFIG["num_experts_per_tok"]) or \
                    routed[:, 0].any() or not routed[:, 1:].any():
                raise ValueError(f"vLLM's routed_experts is {routed.shape} with layer 0 "
                                 f"{'set' if routed[:, 0].any() else 'zero'}; the test reads another layout")
            calls.append({"prompt_ids": list(request.prompt_token_ids), "sampled_ids": sampled,
                          "behavior_log_probs": behavior, "support": support,
                          "routed_experts": routed.astype(int).tolist(), "routed_dtype": str(routed.dtype)})
    import torch

    record = {"vllm": version("vllm"), "gpu": torch.cuda.get_device_name(), "model": MODEL.name,
              "sampling": SAMPLING, "calls": calls}
    references(record)
    OUT.write_text(json.dumps(record) + "\n")
    print(f"wrote {len(calls)} calls to {OUT}")


if __name__ == "__main__":
    main()
