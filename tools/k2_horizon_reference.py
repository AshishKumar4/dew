#!/usr/bin/env python3
"""The K2 Horizon fixtures tests/test_k2_horizon.py checks against, from the
family's own remote code (IFM/K2-Horizon-0.9B at 383cb663, the modeling file
every release ships) on the environment's transformers:

    PYTHONPATH=src:tools .venv-3.12/bin/python tools/k2_horizon_reference.py

tests/fixtures/hf/k2-horizon-tiny/ is a random K2HorizonForCausalLM of two
layers over a width of 16, four query heads of 8 over two key/value heads,
its norms' statistics over two groups (the 3.7B's layernorm_num_groups) and
its rotary the 0.9B's YaRN, with fp32 logits, left-padded logits and the
cached greedy continuation. k2-horizon-{0.9b,3.7b,7b}/ pin the
released configs, which the translation is held to. `--padding` records
instead the rows the release's own token lookup leaves untrained under
k2-horizon-tiny's config with its pad id stated, left out and null
(`reference_source.record_padding`), as padding.json there. `--recording`
records instead k2-horizon-tiny's unpadded and left-padded logits over check
1's seeds (`recorded_references.record_logits`), as recorded_logits.npz there,
which tests/test_hf_reference_decoders.py reads where the remote code cannot
run. `--mova` writes instead k2-horizon-mova-tiny/ from the MoVA release's own
remote code (IFM/K2-Horizon-MoVA-36B-A4B at cca48b66): three layers, the first
dense and the other two routing their MLP to four experts and a shared one and
their values to four value experts, two of each per token, under a biased
sigmoid router scaled by 2.5, every attention output gated by a softplus at
beta log 2, with the same ids, padded rows and greedy continuation, and its
logits over check 1's seeds (recorded_logits.npz).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from hf_reference import write_released_config
from reference_source import config_lookups, hub_modules, record_padding, torch_generated, write_probes
from torch_precision import float64_twin

from dew.interop.verify import scatter_weights

REPO, REVISION = 'IFM/K2-Horizon-0.9B', '383cb66347be'
MOVA_REPO, MOVA_REVISION = 'IFM/K2-Horizon-MoVA-36B-A4B', 'cca48b6631d03c338a0e1d8beb0c392af8becbd3'
RELEASES = {'0.9b': ('IFM/K2-Horizon-0.9B', '383cb66347be'), '3.7b': ('IFM/K2-Horizon-3.7B', 'd6e5432c5963'),
            '7b': ('IFM/K2-Horizon-7B', '3b6ae394c1b8')}
FIXTURES = Path(__file__).resolve().parents[1] / 'tests' / 'fixtures' / 'hf'


def remote(repo: str = REPO, revision: str = REVISION):
    """K2 Horizon's modeling module, imported from the pinned remote code of `repo` as a package."""
    _, modeling = hub_modules(f"k2_horizon_remote_{revision}", repo, revision,
                              ('configuration_k2_horizon.py', 'modeling_k2_horizon.py'))
    return modeling


MOVA = {'num_hidden_layers': 3, 'num_experts': 4, 'num_experts_per_tok': 2, 'moe_intermediate_size': 8,
        'num_shared_experts': 1, 'moe_gate_bias': True, 'router_score_func': 'sigmoid',
        'router_scaling_factor': 2.5, 'norm_topk_prob': True, 'mlp_only_layers': [0],
        'decoder_sparse_step': 1, 'mova_num_experts': 4, 'mova_num_experts_per_tok': 2,
        'attention_gate_func': 'softplus', 'rope_parameters': {'rope_type': 'default', 'rope_theta': 10000.0}}
"""The MoVA fixture's fields over the dense tiny's (`write_tiny`)."""


def write_tiny(modeling, name: str = 'k2-horizon-tiny', fields=None) -> None:
    directory = FIXTURES / name
    directory.mkdir(parents=True, exist_ok=True)
    config = modeling.K2HorizonConfig(**{
        'vocab_size': 64, 'hidden_size': 16, 'intermediate_size': 32, 'num_hidden_layers': 2,
        'num_attention_heads': 4, 'num_key_value_heads': 2, 'head_dim': 8, 'rope_head_dim': 8,
        'max_position_embeddings': 64, 'rms_norm_eps': 1e-6, 'layernorm_num_groups': 2, 'num_experts': 0,
        'mlp_only_layers': [0, 1], 'query_key_norm': False, 'use_sliding_window': False,
        'sliding_window': None, 'tie_word_embeddings': False, 'bos_token_id': 1, 'eos_token_id': None,
        'pad_token_id': 0,
        'rope_parameters': {'rope_type': 'yarn', 'rope_theta': 10000.0, 'factor': 4.0, 'beta_fast': 128.0,
                            'beta_slow': 4.0, 'original_max_position_embeddings': 16, 'truncate': True,
                            'attention_factor': 1.1386294361119891}, **(fields or {})})
    config._attn_implementation = 'eager'
    torch.manual_seed(0)
    model = modeling.K2HorizonForCausalLM(config).float().eval()
    scatter_weights(model, 1234)
    model.save_pretrained(directory, safe_serialization=True)
    write_probes(directory, model, config.vocab_size, torch_generated)


def write_recording(modeling, name: str = 'k2-horizon-tiny', flag: str = '--recording',
                    revision: str = f'{REPO}@{REVISION}') -> None:
    """`name`'s committed weights in the remote code's class, fp32 on eager attention, and their float64
    twin, recorded over check 1's seeds with the rows `write_tiny` padded."""
    import transformers
    from recorded_references import record_logits

    directory = FIXTURES / name
    model = modeling.K2HorizonForCausalLM.from_pretrained(directory, dtype=torch.float32,
                                                          attn_implementation='eager').eval()
    truth = float64_twin(model, lambda: modeling.K2HorizonForCausalLM(model.config))
    mask = np.load(directory / 'attention_mask.npy')
    record_logits(directory, model, truth,
                  f'PYTHONPATH=src:tools .venv-3.12/bin/python tools/k2_horizon_reference.py {flag}',
                  reference='K2HorizonForCausalLM (remote code)', revision=revision,
                  license='Apache-2.0', transformers=transformers.__version__,
                  padded={'attention_mask': mask, 'position_ids': np.maximum(np.cumsum(mask, -1) - 1, 0)})


def write_released() -> None:
    for size, (repo, revision) in RELEASES.items():
        write_released_config(f'k2-horizon-{size}', repo, revision)


if __name__ == '__main__':
    if sys.argv[1:] == ['--padding']:
        modeling = remote()
        record_padding(*config_lookups(lambda fields: modeling.K2HorizonConfig(**fields),
                                       modeling.K2HorizonForCausalLM),
                       json.loads((FIXTURES / 'k2-horizon-tiny' / 'config.json').read_text()),
                       FIXTURES / 'k2-horizon-tiny')
    elif sys.argv[1:] == ['--recording']:
        write_recording(remote())
    elif sys.argv[1:] == ['--mova']:
        modeling = remote(MOVA_REPO, MOVA_REVISION)
        write_tiny(modeling, 'k2-horizon-mova-tiny', MOVA)
        write_recording(modeling, 'k2-horizon-mova-tiny', '--mova', f'{MOVA_REPO}@{MOVA_REVISION}')
    else:
        write_tiny(remote())
        write_released()
