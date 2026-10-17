"""Run in a vLLM environment: the released Dory's greedy continuations and top logprobs, as vLLM
computes them with PR #60149's model, for one cache mode. Writes JSON to argv[2]."""
import json
import sys

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

REPO = "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"
PROMPTS = ["The capital of France is", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
           "Water boils at a temperature of"]
mode, out = sys.argv[1], sys.argv[2]
llm = LLM(model=REPO, dtype="bfloat16", max_model_len=512, enforce_eager=True, gpu_memory_utilization=0.38,
          hf_overrides={"recurrent_kv_cache_mode": mode})
tokenizer = llm.get_tokenizer()
ids = [tokenizer.encode(text) for text in PROMPTS]
params = SamplingParams(temperature=0.0, max_tokens=16, logprobs=8, prompt_logprobs=8)
results = llm.generate([TokensPrompt(prompt_token_ids=row) for row in ids], params)
record = []
for row, result in zip(ids, results, strict=True):
    completion = result.outputs[0]
    record.append({
        "prompt": row, "generated": list(completion.token_ids),
        "step_logprobs": [{str(k): v.logprob for k, v in step.items()} for step in completion.logprobs],
        "prompt_logprobs": [None if step is None else {str(k): v.logprob for k, v in step.items()}
                            for step in result.prompt_logprobs]})
json.dump(record, open(out, "w"))
print("RECORDED", out)
