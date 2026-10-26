"""Generate classic decoder fixtures and check small released checkpoints in fp32.

Run with PYTHONPATH=src. --fixture writes a tiny same-weight reference;
--checkpoint compares the public loader and cached greedy decoding on real weights.
"""

import argparse
import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GPT2Config, OPTConfig

from dew.interop import Pretrained
from dew.interop.verify import _ROUNDING
from dew.sampling import Sampling, generate


def write_fixture(family):
    # The tests import this module as tools.classic_gpt_reference, where
    # hf_reference is not on the path; the fixture runs as a script beside it.
    from hf_reference import write_device_tiny

    config = (OPTConfig(vocab_size=128, hidden_size=32, num_hidden_layers=2,
                        num_attention_heads=4, ffn_dim=48, max_position_embeddings=64,
                        dropout=0, attention_dropout=0, bos_token_id=1,
                        eos_token_id=None, pad_token_id=0) if family == 'opt' else
              GPT2Config(vocab_size=128, n_embd=32, n_layer=2, n_head=4,
                        n_positions=64, n_inner=48, resid_pdrop=0,
                        embd_pdrop=0, attn_pdrop=0, activation_function='gelu_new',
                        bos_token_id=1, eos_token_id=None, pad_token_id=0))
    model = AutoModelForCausalLM.from_config(config, attn_implementation='eager')
    write_device_tiny(f'{family}-tiny', model, 'cuda',
                      f'PYTHONPATH=src python tools/classic_gpt_reference.py --fixture --family {family}')


def check_checkpoint(checkpoint, output, revision):
    """`checkpoint`'s fp32 logits and greedy continuation at its pinned `revision` through the loader
    against transformers'. The reference runs on the CPU whatever JAX's platform, so a CPU-only Torch
    build supplies it beside a GPU and no TF32 enters it."""
    config = AutoConfig.from_pretrained(checkpoint, revision=revision)
    reference = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float32,
                                                    attn_implementation='eager', revision=revision).eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision)
    ids = tokenizer('The capital of France is', return_tensors='np')['input_ids'].astype(np.int32)
    with torch.no_grad():
        expected = reference(torch.from_numpy(ids).long(), use_cache=False).logits.numpy()
        generated = reference.generate(torch.from_numpy(ids).long(), do_sample=False, max_new_tokens=6,
                                       eos_token_id=None, pad_token_id=0).numpy()
    del reference
    loaded = Pretrained.load(checkpoint, dtype='float32', attention_impl='reference', max_seq_len=64,
                             revision=revision)
    loaded = dataclasses.replace(loaded, model=loaded.model.clone(precision=jax.lax.Precision.HIGHEST))
    actual = np.asarray(loaded.model.apply(loaded.variables, jnp.asarray(ids)))
    error = float(np.max(np.abs(expected - actual)))
    # Twice the largest tiny-family error per layer per logit, in fp32 ulps,
    # matches Dew's existing verified-mapping bound (verify._ROUNDING).
    layers = config.num_hidden_layers
    bound = float(2 * _ROUNDING * np.finfo(np.float32).eps * layers * np.max(np.abs(expected)))
    agreement = bool(np.array_equal(actual.argmax(-1), expected.argmax(-1)))
    ours = np.asarray(generate(loaded.model, loaded.variables, ids, 6, key=0,
                               sampling=Sampling(temperature=0)).tokens)
    result = {'checkpoint': checkpoint, 'revision': revision,
              'dtype': 'float32', 'precision': 'highest',
              'transformers': transformers.__version__, 'jax': jax.__version__,
              'torch': torch.__version__, 'reference_attention': 'eager', 'tf32': False,
              'device': jax.devices()[0].device_kind, 'max_abs_error': error, 'bound': bound,
              'argmax_agreement': agreement, 'generated': generated.tolist(),
              'generation_agreement': bool(np.array_equal(ours, generated))}
    assert error <= bound and agreement and result['generation_agreement'], result
    if output:
        Path(output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', action='store_true')
    parser.add_argument('--family', choices=('gpt2', 'opt'), default='gpt2')
    parser.add_argument('--checkpoint')
    parser.add_argument('--revision', help='the commit --checkpoint is pinned at')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.fixture:
        write_fixture(args.family)
    if args.checkpoint:
        if args.revision is None:
            parser.error('--checkpoint is checked at a pinned --revision')
        check_checkpoint(args.checkpoint, args.output, args.revision)


if __name__ == '__main__':
    main()
