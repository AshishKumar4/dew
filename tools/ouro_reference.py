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
loaded checkpoint to.
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import transformers
from diffusers_wan_reference import float64
from huggingface_hub import hf_hub_download

from dew.interop.verify import probe_ids, scatter_weights

REPO, REVISION = 'ByteDance/Ouro-1.4B', '574fa66cb8bf5abdc979642d01cf2b79b16bfab1'
FIXTURES = Path(__file__).resolve().parents[1] / 'tests' / 'fixtures' / 'hf'
COLUMNS = np.random.RandomState(11).choice(49152, 64, replace=False)


def remote():
    """Ouro's modeling module, imported from the pinned remote code as a package."""
    package = Path(tempfile.mkdtemp()) / 'ouro_remote'
    package.mkdir()
    (package / '__init__.py').write_text('')
    for name in ('configuration_ouro.py', 'modeling_ouro.py'):
        shutil.copy(hf_hub_download(REPO, name, revision=REVISION), package / name)
    sys.path.insert(0, str(package.parent))
    import ouro_remote.modeling_ouro as modeling
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
    mask = np.ones_like(ids, bool)
    mask[1, :3] = False
    padded = np.where(mask, ids, 0)
    positions = np.maximum(np.cumsum(mask, axis=-1) - 1, 0)
    arrays = {'input_ids': ids, 'padded_ids': padded, 'attention_mask': mask}
    arrays['logits'], arrays['exit_logits'] = outputs(model, ids)
    arrays['padded_logits'], _ = outputs(model, padded, mask, positions)
    with torch.no_grad():
        arrays['generated'] = model.generate(torch.from_numpy(ids[:, :4]).long(), do_sample=False,
                                             max_new_tokens=6, eos_token_id=None, pad_token_id=0).numpy()
    with float64():
        # Built inside the widening, so the rotary table is computed in
        # float64 rather than widened from fp32; the weights are the same.
        truth = modeling.OuroForCausalLM(config).double()
        truth.load_state_dict(model.state_dict())
        truth = last_pass(truth)
        arrays['logits_f64'], arrays['exit_logits_f64'] = outputs(truth, ids)
        arrays['padded_logits_f64'], _ = outputs(truth, padded, mask, positions)
    for name, array in arrays.items():
        np.save(directory / f'{name}.npy', array)
    print(f"{directory}: {sorted(path.name for path in directory.iterdir())}")


def write_released(modeling, full: bool) -> None:
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
    assert transformers.__version__ == '4.56.2', f"transformers {transformers.__version__} is on the path"
    modeling = remote()
    write_tiny(modeling)
    write_released(modeling, parser.parse_args().full)
