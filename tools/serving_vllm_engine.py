"""vLLM's engine in-process for `benchmark_lm_serving`'s vllm-engine backend, run in vLLM's own interpreter
(the fleet's DEW_VLLM_PYTHON), the bench's clocks and sweep its own."""

from __future__ import annotations

import os
import uuid

from benchmark_lm_serving import engine_sweep, stamped, vllm_parallelism

COHORT = "vllm"
"""The releases this file runs against, which its type check reads (tools/reference_releases.RELEASES)."""


async def engine_points(args):
    """The same scheduler/stream clocks, without HTTP serialization. Output stays token ids, not text, as
    Dew's and SGLang's do (`detokenize=False`). An engine over several GPUs starts its workers by
    spawning them, as vLLM advises once the parent has touched CUDA, which forked workers cannot
    initialize again."""
    parallelism = vllm_parallelism(args.mesh)
    if parallelism.get("tensor_parallel_size", 1) > 1:
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.inputs import TokensPrompt
    from vllm.v1.engine.async_llm import AsyncLLM

    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
        model=args.model, dtype="bfloat16", max_model_len=args.prompt + args.output,
        max_num_seqs=max(args.slots), gpu_memory_utilization=0.90,
        generation_config="vllm", enable_prefix_caching=False, disable_log_stats=True, **parallelism))

    async def streamed(prompt, output):
        """The time each of a request's tokens reached the client."""
        policy = SamplingParams(temperature=0, top_p=1, top_k=-1, min_p=0, repetition_penalty=1,
                                max_tokens=output, ignore_eos=True, detokenize=False)
        return await stamped(engine.generate(TokensPrompt(prompt_token_ids=prompt.tolist()), policy,
                                             uuid.uuid4().hex),
                             lambda result: len(result.outputs[0].token_ids))

    return await engine_sweep(args, streamed, engine)
