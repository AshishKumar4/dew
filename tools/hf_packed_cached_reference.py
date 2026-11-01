#!/usr/bin/env python3
"""transformers' own packed rows and cached decoding, which tests/test_decoder_packed_cached_source.py
runs in the test over drawn ids on the committed llama-tiny (grouped heads, projection biases) and
mistral-tiny (a sliding window of 4) checkpoints, its `LlamaForCausalLM` and `MistralForCausalLM` on
eager attention, and their float64 twins:

- packed rows (`packed`): several documents to a row, each restarting
  `position_ids` at zero and no attention mask, from which transformers builds
  its own packed mask (`masking_utils.find_packed_sequence_indices`), each
  document's logits the ones it gets alone;
- cached decoding (`cached`): a prompt prefilled into transformers' cache,
  then one token a step, past the sliding window, the logits of the prompt's
  last position and of every step, which are the full sequence's.

The packed call asks for no cache: transformers looks for packing only when it builds none.
"""

import numpy as np
import torch

# Documents per packed row, each row 16 tokens.
PACKING = ((5, 7, 4), (9, 7))
PROMPT, STEPS = 5, 9


def packed(net, ids: np.ndarray) -> torch.Tensor:
    """The packed rows' logits, each document's positions from zero (`PACKING`)."""
    positions = torch.tensor([[step for length in row for step in range(length)] for row in PACKING])
    return net(input_ids=torch.as_tensor(ids), position_ids=positions, use_cache=False).logits


def cached(net, ids: np.ndarray) -> torch.Tensor:
    """The prompt's last logits and each step's, through transformers' cache."""
    output = net(input_ids=torch.as_tensor(ids[:, :PROMPT]), use_cache=True)
    cache, steps = output.past_key_values, [output.logits[:, -1]]
    for position in range(PROMPT, PROMPT + STEPS):
        output = net(input_ids=torch.as_tensor(ids[:, position:position + 1]), past_key_values=cache,
                     use_cache=True)
        cache = output.past_key_values
        steps.append(output.logits[:, -1])
    return torch.stack(steps, dim=1)
