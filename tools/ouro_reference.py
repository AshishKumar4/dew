#!/usr/bin/env python3
"""The Ouro fixtures tests/test_ouro.py checks against, from Ouro's own remote
code (ByteDance/Ouro-1.4B at 574fa66c) on transformers 4.56.2. 5.16.1 no
longer has the `ROPE_INIT_FUNCTIONS['default']` the code calls, and before 4.56
transformers' Cache refuses the cache fix it carries (its card still says
4.54.1). So 4.56.2 is installed apart, for the environment's Python, and put
first on the path:

    uv pip install --python .venv-3.12/bin/python --target /tmp/tf456 --no-deps \\
        transformers==4.56.2 tokenizers==0.22.1 huggingface_hub==0.35.3
    PYTHONPATH=/tmp/tf456:src:tools .venv-3.12/bin/python tools/ouro_reference.py [--full]

tests/fixtures/hf/ouro-tiny/ is a random OuroForCausalLM of two layers over a
width of 16, four query heads of 8 over two key/value heads, its stack run
three times, with fp32 and float64 logits, left-padded logits, the cached
greedy continuation and each pass's exit-gate logits. ouro-1.4b/ pins the
released config; `--full` adds probe.npz, the released model's argmax and 64
of its logit columns on the probe ids, which the network test holds the
loaded checkpoint to. `--padding` records instead the rows the release's own
token lookup leaves untrained under ouro-tiny's config with its pad id stated,
left out and null (`reference_source.record_padding`), as padding.json there.
`--recording` records instead ouro-tiny's unpadded and left-padded logits over
check 1's seeds (`recorded_references.record_logits`), as recorded_logits.npz
there, which tests/test_hf_reference_decoders.py reads where the remote code
cannot run.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from reference_source import config_lookups, hub_modules, padded_probe, record_padding
from torch_precision import float64, float64_twin

REPO, REVISION = 'ByteDance/Ouro-1.4B', '574fa66cb8bf5abdc979642d01cf2b79b16bfab1'
FIXTURES = Path(__file__).resolve().parents[1] / 'tests' / 'fixtures' / 'hf'
COLUMNS = np.random.RandomState(11).choice(49152, 64, replace=False)


def remote():
    """Ouro's modeling module, imported from the pinned remote code as a package."""
    _, modeling = hub_modules('ouro_remote', REPO, REVISION, ('configuration_ouro.py', 'modeling_ouro.py'))
    return modeling


def last_pass(model):
    """`model` reading its last pass, as the threshold of 1.0 does unless a hazard rounds to 1."""
    model.early_exit_step = model.config.total_ut_steps - 1
    return model.eval()


def outputs(model, ids, mask=None, positions=None):
    """The logits and the stacked exit-gate logits of one uncached call."""
    inputs = {'input_ids': torch.from_numpy(ids).long(), 'use_cache': False}
    if mask is not None:
        inputs.update(attention_mask=torch.from_numpy(mask), position_ids=torch.from_numpy(positions).long())
    with torch.no_grad():
        logits = model(**inputs).logits
        _, _, gates = model.model(**inputs)
    return logits.numpy(), torch.stack(gates).numpy()


def write_tiny(modeling) -> None:
    from dew.interop.verify import probe_ids, scatter_weights

    directory = FIXTURES / 'ouro-tiny'
    directory.mkdir(parents=True, exist_ok=True)
    config = modeling.OuroConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, max_position_embeddings=64, rms_norm_eps=1e-6, rope_theta=10000.0,
        tie_word_embeddings=False, total_ut_steps=3, bos_token_id=1, eos_token_id=None, pad_token_id=0)
    config._attn_implementation = 'eager'
    torch.manual_seed(0)
    model = modeling.OuroForCausalLM(config).float()
    scatter_weights(model, 1234)
    model = last_pass(model)
    model.save_pretrained(directory, safe_serialization=True)
    ids = probe_ids(config.vocab_size)
    mask, padded, positions = padded_probe(ids)
    arrays = {'input_ids': ids, 'attention_mask': mask}
    arrays['logits'], arrays['exit_logits'] = outputs(model, ids)
    arrays['padded_logits'], _ = outputs(model, padded, mask, positions)
    with torch.no_grad():
        arrays['generated'] = model.generate(torch.from_numpy(ids[:, :4]).long(), do_sample=False,
                                             max_new_tokens=6, eos_token_id=None, pad_token_id=0).numpy()
    # Its float64 twin (`float64_twin`): built inside the widening, so the rotary table is computed in
    # float64 rather than widened from fp32; the weights are the same.
    truth = last_pass(float64_twin(model, lambda: modeling.OuroForCausalLM(config)))
    with float64():
        arrays['logits_f64'], arrays['exit_logits_f64'] = outputs(truth, ids)
    for name, array in arrays.items():
        np.save(directory / f'{name}.npy', array)
    print(f"{directory}: {sorted(path.name for path in directory.iterdir())}")


def write_recording(modeling) -> None:
    """ouro-tiny's committed weights in the remote code's class, fp32 on eager attention reading its last
    pass, and their float64 twin, recorded over check 1's seeds with the rows `write_tiny` padded."""
    import transformers
    from recorded_references import record_logits

    directory = FIXTURES / 'ouro-tiny'
    model = last_pass(modeling.OuroForCausalLM.from_pretrained(directory, torch_dtype=torch.float32,
                                                               attn_implementation='eager'))
    truth = last_pass(float64_twin(model, lambda: modeling.OuroForCausalLM(model.config)))
    mask = np.load(directory / 'attention_mask.npy')
    record_logits(directory, model, truth,
                  'PYTHONPATH=/tmp/tf456:src:tools .venv-3.12/bin/python tools/ouro_reference.py --recording',
                  reference='OuroForCausalLM (remote code)', revision=f'{REPO}@{REVISION}',
                  license='Apache-2.0', transformers=transformers.__version__, planned=False,
                  padded={'attention_mask': mask, 'position_ids': np.maximum(np.cumsum(mask, -1) - 1, 0)})


def write_released(modeling, full: bool) -> None:
    from dew.interop.verify import probe_ids

    directory = FIXTURES / 'ouro-1.4b'
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copy(hf_hub_download(REPO, 'config.json', revision=REVISION), directory / 'config.json')
    (directory / 'source.json').write_text(f'{{"repo": "{REPO}", "revision": "{REVISION}"}}\n')
    if not full:
        return
    model = last_pass(modeling.OuroForCausalLM.from_pretrained(
        REPO, revision=REVISION, torch_dtype=torch.float32, attn_implementation='eager'))
    ids = probe_ids(model.config.vocab_size)
    logits, _ = outputs(model, ids)
    np.savez(directory / 'probe.npz', input_ids=ids, argmax=logits.argmax(-1), columns=COLUMNS,
             logits=logits[..., COLUMNS])
    print(f"{directory / 'probe.npz'}: argmax {logits.argmax(-1).tolist()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help="also run the released model for probe.npz")
    parser.add_argument('--padding', action='store_true', help="record the rows the lookup leaves untrained")
    parser.add_argument('--recording', action='store_true', help="record ouro-tiny's logits over the seeds")
    arguments = parser.parse_args()
    from reference_releases import require

    require("ouro")
    modeling = remote()
    if arguments.padding:
        tiny = FIXTURES / 'ouro-tiny'
        lookups = config_lookups(lambda fields: modeling.OuroConfig(**fields), modeling.OuroForCausalLM)
        record_padding(*lookups, json.loads((tiny / 'config.json').read_text()), tiny)
    elif arguments.recording:
        write_recording(modeling)
    else:
        write_tiny(modeling)
        write_released(modeling, arguments.full)
