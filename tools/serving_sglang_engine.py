"""SGLang's engine in-process for `benchmark_lm_serving`'s sglang-engine backend, run in SGLang's own
interpreter (the fleet's DEW_SGLANG_PYTHON), the bench's clocks and sweep its own."""

from __future__ import annotations


from benchmark_lm_serving import engine_sweep, stamped

COHORT = "sglang"
"""The releases this file runs against, which its type check reads (tools/reference_releases.RELEASES)."""


async def sglang_points(args):
    """SGLang's engine in-process, on the same clocks as vllm-engine, its output token ids."""
    import sglang

    engine = sglang.Engine(model_path=args.model, dtype="bfloat16", context_length=args.prompt + args.output,
                           max_running_requests=max(args.slots), mem_fraction_static=0.90,
                           disable_radix_cache=True, skip_tokenizer_init=True, log_level="error")
    policy = {"temperature": 0, "top_p": 1, "top_k": -1, "min_p": 0, "repetition_penalty": 1,
              "ignore_eos": True}

    async def streamed(prompt, output):
        """The time each of a request's tokens reached the client, from SGLang's stream of growing output."""
        stream = await engine.async_generate(input_ids=prompt.tolist(), stream=True,
                                             sampling_params={**policy, "max_new_tokens": output})
        return await stamped(stream, lambda chunk: chunk["meta_info"]["completion_tokens"])

    return await engine_sweep(args, streamed, engine)
