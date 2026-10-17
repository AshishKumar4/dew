"""Run in Dew's environment: the released Dory under Dew, teacher-forced over vLLM's continuation
and decoding with each cache, against what vLLM recorded."""
import dataclasses
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

from dew.interop import Pretrained
from dew.nn.backbones.causal_transformer import CausalTransformer

REPO = "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"
loaded = Pretrained.load(REPO, dtype="bfloat16", attention_impl="xla")
report = {"device": jax.devices()[0].device_kind}
for mode, path in (("per_loop", sys.argv[1]), ("last_loop", sys.argv[2])):
    recorded = json.load(open(path))
    model = loaded.model.clone(loop=dataclasses.replace(loaded.model.loop, shared_cache=mode == "last_loop"))
    model = model.with_cache_capacity(128)
    rows = []
    for row in recorded:
        prompt, generated = row["prompt"], row["generated"]
        full = jnp.asarray([prompt + generated], jnp.int32)
        forced = np.asarray(jax.nn.log_softmax(model.apply(loaded.variables, full)[0].astype(jnp.float32)))
        # Teacher-forced: Dew's logprob of each id vLLM ranked, at each generated step.
        diffs, agree, margins = [], 0, []
        for step, ranked in enumerate(row["step_logprobs"]):
            at = len(prompt) - 1 + step
            for token, logprob in ranked.items():
                diffs.append(forced[at, int(token)] - logprob)
            best = sorted(ranked.values(), reverse=True)
            margins.append(best[0] - best[1] if len(best) > 1 else 99.0)
            agree += int(forced[at].argmax() == generated[step])
        # Cached greedy decoding under this mode.
        cache = model.apply(loaded.variables, 1, method=CausalTransformer.init_cache, mutable=["cache"])[1]["cache"]
        logits, mutated = model.apply({**loaded.variables, "cache": cache}, jnp.asarray([prompt]), decode=True,
                                      mutable=["cache"])
        decoded, step_diffs = [], []
        for step in range(len(generated)):
            last = np.asarray(jax.nn.log_softmax(logits[0, -1].astype(jnp.float32)))
            for token, logprob in row["step_logprobs"][step].items():
                step_diffs.append(last[int(token)] - logprob)
            decoded.append(int(last.argmax()))
            logits, mutated = model.apply({**loaded.variables, "cache": mutated["cache"]},
                                          jnp.asarray([[generated[step]]]), decode=True, mutable=["cache"])
        rows.append({"forced max|diff|": float(np.max(np.abs(diffs))), "forced mean|diff|": float(np.mean(np.abs(diffs))),
                     "forced argmax agrees": f"{agree}/{len(generated)}", "min vllm margin": float(min(margins)),
                     "cached max|diff|": float(np.max(np.abs(step_diffs))),
                     "cached mean|diff|": float(np.mean(np.abs(step_diffs))),
                     "cached tokens agree": f"{sum(a == b for a, b in zip(decoded, generated))}/{len(generated)}",
                     "first divergence": next((i for i, (a, b) in enumerate(zip(decoded, generated)) if a != b), None)})
    report[mode] = rows
print("DEWSIDE " + json.dumps(report))
