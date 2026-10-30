#!/usr/bin/env python3
"""The MiniMax fixtures tests/test_minimax.py and tests/test_hf_reference_decoders.py check against, from
the release's own remote code (MiniMaxAI/MiniMax-Text-01 at a7351bf2, whose modeling file MiniMax-M1's
repeats but for its class names) on the environment's transformers:

    PYTHONPATH=src:tools .venv-3.12/bin/python tools/minimax_reference.py

tests/fixtures/hf/minimax-tiny/ is a random MiniMaxText01ForCausalLM of four layers over a width of 16,
lightning, lightning, softmax and lightning attention (`attn_type_list`), six query heads of 8 over two
key/value heads (six heads take ALiBi's slopes past a power of two), the softmax layer turning half of
each head, four experts two a token, and the releases' postnorm block and factors, with fp32 logits,
left-padded logits and the greedy continuation. Its `block_size` of 4, which the remote code does not
read (its blocks are 256 tokens, :51), runs Dew's recurrence in chunks of 4. `--recording` records
instead its unpadded and left-padded logits over check 1's seeds (`recorded_references.record_logits`),
as recorded_logits.npz there, which the tests read where the remote code cannot run. minimax-text-01
and minimax-text-01-hf pin the released config in the release's own spelling and in transformers'.

Three things of transformers 5 and of a CPU stand in for what the remote code was written against,
and one mends its padded rows: transformers 5 removed `is_torch_fx_available`, which the module
reads only to wrap a mask helper for FX tracing, so it reads False; it reads a model's tied weights
as a mapping of each to its source,
where the remote code lists them (`_tied_weights_keys`, :1405), so the head's tie is stated as a
mapping to the embedding, which ties nothing at `tie_word_embeddings` false; and the softmax layers
run flash attention alone (`MiniMaxText01FlashAttention2`, :576-760, built for every softmax layer,
:993-1000), which needs CUDA, so `flash` computes what its `flash_attn_varlen_func` does over the
unpadded rows: causal attention at its default scale over a row's real keys, a padded query's row
zero (`pad_input`). The decoder layer hands every attention its mask as `attn_mask` (:1028-1036),
the lightning layers' keyword, which the flash layers' `forward` takes into its `**kwargs` and drops,
so the release's softmax layers attend a left-padded row's padding; `masked` hands them the mask as
`attention_mask`, the padding left out as the lightning layers leave it out and as transformers'
MiniMax masks its softmax layers (modeling_minimax.py:660-690). The config class names its type
`MiniMaxText01`, which the fixture's config.json states as the release's does, `minimax_text_01`.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from hf_reference import write_released_config
from reference_source import hub_modules, write_probes
from torch_precision import float64_twin

from dew.interop.verify import scatter_weights

REPO, REVISION = 'MiniMaxAI/MiniMax-Text-01', 'a7351bf2bee0e1253919d349f1ad304e6dac13e9'
RELEASES = {'minimax-text-01': (REPO, REVISION),
            'minimax-text-01-hf': ('MiniMaxAI/MiniMax-Text-01-hf',
                                   'f7ce01366e8585a8948f19aedc8e20628c6965e5')}
"""The released configs the fixtures pin; MiniMax-M1-80k-hf's (3dbbd8d1) is Text-01-hf's, key for key."""
FIXTURES = Path(__file__).resolve().parents[1] / 'tests' / 'fixtures' / 'hf'
ALPHA = 3.5565588200778455
"""Every release's alpha factor."""
TINY = {
    'vocab_size': 64, 'hidden_size': 16, 'intermediate_size': 32, 'num_hidden_layers': 4,
    'num_attention_heads': 6, 'num_key_value_heads': 2, 'head_dim': 8, 'rotary_dim': 4,
    'attn_type_list': [0, 0, 1, 0], 'num_local_experts': 4, 'num_experts_per_tok': 2,
    'max_position_embeddings': 64, 'rms_norm_eps': 1e-5, 'rope_theta': 10000.0, 'postnorm': True,
    **{f'layernorm_{site}_alpha': ALPHA for site in ('full_attention', 'linear_attention', 'mlp')},
    **{f'layernorm_{site}_beta': 1.0 for site in ('full_attention', 'linear_attention', 'mlp')},
    'shared_intermediate_size': 0, 'block_size': 4, 'tie_word_embeddings': False, 'bos_token_id': 1,
    'eos_token_id': None, 'pad_token_id': 0}
"""minimax-tiny's MiniMaxText01Config."""


def flash(self, query, key, value, attention_mask, query_length, dropout=0.0, softmax_scale=None,
          use_sliding_windows=False):
    """`_flash_attention_forward` (:681-760) as flash attention computes it, `[B, S, H, D]` in and out."""
    heads, width = query.shape[2], query.shape[3]
    query = query.transpose(1, 2)
    key, value = (x.transpose(1, 2).repeat_interleave(heads // x.shape[2], dim=1) for x in (key, value))
    scores = query @ key.transpose(-1, -2) * (width ** -0.5 if softmax_scale is None else softmax_scale)
    seen = torch.ones(query.shape[2], key.shape[2], dtype=torch.bool).tril(key.shape[2] - query.shape[2])
    if attention_mask is not None:
        real = attention_mask.bool()
        seen = seen[None, None] & real[:, None, None, :]
    weights = torch.softmax(scores.masked_fill(~seen, -math.inf), dim=-1).nan_to_num()
    out = (weights @ value).transpose(1, 2)
    return out if attention_mask is None else out * attention_mask[:, :, None, None].to(out.dtype)


def masked(forward):
    """A flash layer's `forward` that takes the mask the decoder layer passes as `attn_mask`."""
    def run(self, hidden_states, attention_mask=None, *args, attn_mask=None, **kwargs):
        return forward(self, hidden_states, attn_mask if attention_mask is None else attention_mask, *args,
                       **kwargs)
    return run


def remote():
    """The release's configuration and modeling modules, imported from its pinned remote code as a
    package, with `is_torch_fx_available` and flash attention stood in for (the module docstring)."""
    import transformers.utils.import_utils as import_utils

    if not hasattr(import_utils, 'is_torch_fx_available'):
        import_utils.is_torch_fx_available = lambda: False
    configuration, modeling = hub_modules(f"minimax_remote_{REVISION[:8]}", REPO, REVISION,
                                          ('configuration_minimax_text_01.py', 'modeling_minimax_text_01.py'))
    modeling._flash_supports_window_size = False
    modeling.MiniMaxText01ForCausalLM._tied_weights_keys = {'lm_head.weight': 'model.embed_tokens.weight'}
    modeling.MiniMaxText01FlashAttention2._flash_attention_forward = flash
    modeling.MiniMaxText01FlashAttention2.forward = masked(modeling.MiniMaxText01FlashAttention2.forward)
    return configuration, modeling


def write_tiny(configuration, modeling) -> None:
    directory = FIXTURES / 'minimax-tiny'
    directory.mkdir(parents=True, exist_ok=True)
    config = configuration.MiniMaxText01Config(**TINY)
    torch.manual_seed(0)
    model = modeling.MiniMaxText01ForCausalLM(config).float().eval()
    scatter_weights(model, 1234)
    model.save_pretrained(directory, safe_serialization=True)
    written = json.loads((directory / 'config.json').read_text())
    written['model_type'] = 'minimax_text_01'
    (directory / 'config.json').write_text(json.dumps(written, indent=2) + '\n')
    write_probes(directory, model, config.vocab_size)


def integer_mask(model):
    """`model` called with a tokenizer's integer mask where a caller passes booleans: the lightning
    layers subtract it from one (:281), which torch refuses of a boolean tensor."""
    def run(attention_mask=None, **inputs):
        mask = {} if attention_mask is None else {'attention_mask': attention_mask.long()}
        return model(**inputs, **mask)
    return run


def write_recording(configuration, modeling) -> None:
    """minimax-tiny's committed weights in the remote code's class, fp32, and their float64 twin,
    recorded over check 1's seeds with the rows `write_tiny` padded. The class is built and the weights
    loaded into it, as `from_pretrained` would build it on the meta device, where the slopes the model
    keeps as a plain tensor (:1221) stay without values."""
    import transformers
    from recorded_references import record_logits
    from safetensors.torch import load_file

    directory = FIXTURES / 'minimax-tiny'
    model = modeling.MiniMaxText01ForCausalLM(configuration.MiniMaxText01Config.from_pretrained(directory))
    model.load_state_dict(load_file(directory / 'model.safetensors'), strict=True)
    model = model.float().eval()
    truth = float64_twin(model, lambda: modeling.MiniMaxText01ForCausalLM(model.config))
    mask = np.load(directory / 'attention_mask.npy')
    record_logits(directory, integer_mask(model), integer_mask(truth),
                  'PYTHONPATH=src:tools .venv-3.12/bin/python tools/minimax_reference.py --recording',
                  reference='MiniMaxText01ForCausalLM (remote code, flash attention run eagerly)',
                  revision=f'{REPO}@{REVISION}', license='MIT', transformers=transformers.__version__,
                  padded={'attention_mask': mask, 'position_ids': np.maximum(np.cumsum(mask, -1) - 1, 0)})


def write_released() -> None:
    for name, (repo, revision) in RELEASES.items():
        write_released_config(name, repo, revision)


if __name__ == '__main__':
    if sys.argv[1:] == ['--recording']:
        write_recording(*remote())
    else:
        write_tiny(*remote())
        write_released()
