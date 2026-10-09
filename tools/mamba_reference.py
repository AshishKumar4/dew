#!/usr/bin/env python3
"""The Mamba fixture tests/test_mamba.py checks against, from transformers
5.16.1 on CPU.

    PYTHONPATH=src:tools .venv/bin/python tools/mamba_reference.py

tests/fixtures/hf/mamba-tiny/ is a random `MambaForCausalLM` in the HF
layout: two selective-scan layers of 16 inner channels over a width of 8, a
state of 4 per channel from a rank-2 step, the conv biased, and an untied
head, with tools/hf_reference.py's fp32 and float64
logits, left-padded logits and cached greedy continuation
(`write_classic_tiny`). The projections are bias-free, as every released
Mamba's are: the reference zeroes a padded slot's input before `in_proj`,
so a biased one feeds its bias into the first real tokens' conv window,
which an unpadded row never reads. mamba-130m-hf/ pins the released config.
"""

from hf_reference import write_classic_tiny, write_released_config
from transformers import MambaConfig, MambaForCausalLM

MAMBA_CONFIG = ('mamba-130m-hf', 'state-spaces/mamba-130m-hf', '1e76775f628fbf1350fbe4dbb3d971ba64af25a1')


def tiny_mamba() -> MambaForCausalLM:
    import torch

    torch.manual_seed(0)
    return MambaForCausalLM(MambaConfig(
        vocab_size=64, hidden_size=8, expand=2, state_size=4, time_step_rank=2, conv_kernel=4,
        num_hidden_layers=2, layer_norm_epsilon=3e-5, use_bias=False, use_conv_bias=True,
        tie_word_embeddings=False, bos_token_id=1, eos_token_id=None, pad_token_id=0))


if __name__ == '__main__':
    write_classic_tiny('mamba-tiny', tiny_mamba())
    write_released_config(*MAMBA_CONFIG)
