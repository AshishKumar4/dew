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
run.
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from reference_source import config_lookups, hub_modules, padded_probe, record_padding
from torch_precision import float64_twin

from dew.interop.verify import probe_ids, scatter_weights

REPO, REVISION = 'IFM/K2-Horizon-0.9B', '383cb66347be'
RELEASES = {'0.9b': ('IFM/K2-Horizon-0.9B', '383cb66347be'), '3.7b': ('IFM/K2-Horizon-3.7B', 'd6e5432c5963'),
            '7b': ('IFM/K2-Horizon-7B', '3b6ae394c1b8')}
FIXTURES = Path(__file__).resolve().parents[1] / 'tests' / 'fixtures' / 'hf'


def remote():
    """K2 Horizon's modeling module, imported from the pinned remote code as a package."""
    _, modeling = hub_modules('k2_horizon_remote', REPO, REVISION,
                              ('configuration_k2_horizon.py', 'modeling_k2_horizon.py'))
    return modeling


def logits(model, ids, mask=None, positions=None):
    inputs = {'input_ids': torch.from_numpy(ids).long(), 'use_cache': False}
    if mask is not None:
        inputs.update(attention_mask=torch.from_numpy(mask), position_ids=torch.from_numpy(positions).long())
    with torch.no_grad():
        return model(**inputs).logits.numpy()


def write_tiny(modeling) -> None:
    directory = FIXTURES / 'k2-horizon-tiny'
    directory.mkdir(parents=True, exist_ok=True)
    config = modeling.K2HorizonConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, rope_head_dim=8, max_position_embeddings=64, rms_norm_eps=1e-6,
        layernorm_num_groups=2, num_experts=0, mlp_only_layers=[0, 1], query_key_norm=False,
        use_sliding_window=False, sliding_window=None, tie_word_embeddings=False, bos_token_id=1,
        eos_token_id=None, pad_token_id=0,
        rope_parameters={'rope_type': 'yarn', 'rope_theta': 10000.0, 'factor': 4.0, 'beta_fast': 128.0,
                         'beta_slow': 4.0, 'original_max_position_embeddings': 16, 'truncate': True,
                         'attention_factor': 1.1386294361119891})
    config._attn_implementation = 'eager'
    torch.manual_seed(0)
    model = modeling.K2HorizonForCausalLM(config).float().eval()
    scatter_weights(model, 1234)
    model.save_pretrained(directory, safe_serialization=True)
    ids = probe_ids(config.vocab_size)
    mask, padded, positions = padded_probe(ids)
    arrays = {'input_ids': ids, 'attention_mask': mask,
              'logits': logits(model, ids), 'padded_logits': logits(model, padded, mask, positions)}
    with torch.no_grad():
        arrays['generated'] = model.generate(torch.from_numpy(ids[:, :4]).long(), do_sample=False,
                                             max_new_tokens=6, eos_token_id=None, pad_token_id=0).numpy()
    for name, array in arrays.items():
        np.save(directory / f'{name}.npy', array)
    print(f"{directory}: {sorted(path.name for path in directory.iterdir())}")


def write_recording(modeling) -> None:
    """k2-horizon-tiny's committed weights in the remote code's class, fp32 on eager attention, and their
    float64 twin, recorded over check 1's seeds with the rows `write_tiny` padded."""
    import transformers
    from recorded_references import record_logits

    directory = FIXTURES / 'k2-horizon-tiny'
    model = modeling.K2HorizonForCausalLM.from_pretrained(directory, dtype=torch.float32,
                                                          attn_implementation='eager').eval()
    truth = float64_twin(model, lambda: modeling.K2HorizonForCausalLM(model.config))
    mask = np.load(directory / 'attention_mask.npy')
    record_logits(directory, model, truth,
                  'PYTHONPATH=src:tools .venv-3.12/bin/python tools/k2_horizon_reference.py --recording',
                  reference='K2HorizonForCausalLM (remote code)', revision=f'{REPO}@{REVISION}',
                  license='Apache-2.0', transformers=transformers.__version__,
                  padded={'attention_mask': mask, 'position_ids': np.maximum(np.cumsum(mask, -1) - 1, 0)})


def write_released() -> None:
    for size, (repo, revision) in RELEASES.items():
        directory = FIXTURES / f'k2-horizon-{size}'
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copy(hf_hub_download(repo, 'config.json', revision=revision), directory / 'config.json')
        (directory / 'source.json').write_text(f'{{"repo": "{repo}", "revision": "{revision}"}}\n')


if __name__ == '__main__':
    if sys.argv[1:] == ['--padding']:
        modeling = remote()
        record_padding(*config_lookups(lambda fields: modeling.K2HorizonConfig(**fields),
                                       modeling.K2HorizonForCausalLM),
                       json.loads((FIXTURES / 'k2-horizon-tiny' / 'config.json').read_text()),
                       FIXTURES / 'k2-horizon-tiny')
    elif sys.argv[1:] == ['--recording']:
        write_recording(remote())
    else:
        write_tiny(remote())
        write_released()
