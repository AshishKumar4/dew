"""A vLLM logits processor that draws the tokens a request names, for tools/dory_vllm_reference.py's
residual orders: the engine decodes a recorded continuation again, and its raw log-probabilities,
taken before any processor, are each step's own. Loaded by its name, `dory_vllm_forcing:Forced`, in
vLLM's environment with tools/ on the path; it imports vLLM's model runner interface (V2) at once.
"""

from __future__ import annotations

import torch
from vllm.v1.worker.gpu.sample.logits_processor.interface import LogitsProcessor


class Forced(LogitsProcessor):
    """Each step of a request whose `extra_args` name `forced` tokens and its `prompt` length draws
    that step's token: every other logit of its row is -inf."""

    def __init__(self, vllm_config, req_states):
        self.forced: dict[int, tuple[list[int], int]] = {}

    def add_request(self, req_idx: int, sampling_params) -> bool:
        given = sampling_params.extra_args or {}
        if 'forced' not in given:
            self.forced.pop(req_idx, None)
            return False
        self.forced[req_idx] = (list(given['forced']), int(given['prompt']))
        return True

    def apply(self, logits: torch.Tensor, ctx) -> torch.Tensor:
        for row, (slot, position) in enumerate(zip(ctx.expanded_idx_mapping.tolist(), ctx.pos.tolist(),
                                                   strict=True)):
            if slot in self.forced:
                tokens, prompt = self.forced[slot]
                token = tokens[position - prompt + 1]
                logits[row].fill_(-torch.inf)
                logits[row, token] = 0.0
        return logits
