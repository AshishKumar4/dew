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
from dew.interop.verify import _ROUNDING, probe_ids, scatter_weights

FIXTURES = Path(__file__).resolve().parents[1] / 'tests' / 'fixtures' / 'hf'


def write_fixture(family):
    config = (OPTConfig(vocab_size=128, hidden_size=32, num_hidden_layers=2,
                        num_attention_heads=4, ffn_dim=48, max_position_embeddings=64,
                        dropout=0, attention_dropout=0, bos_token_id=1,
                        eos_token_id=None, pad_token_id=0) if family == 'opt' else
              GPT2Config(vocab_size=128, n_embd=32, n_layer=2, n_head=4,
                        n_positions=64, n_inner=48, resid_pdrop=0,
                        embd_pdrop=0, attn_pdrop=0, activation_function='gelu_new',
                        bos_token_id=1, eos_token_id=None, pad_token_id=0))
    model = AutoModelForCausalLM.from_config(config, attn_implementation='eager')
    scatter_weights(model)
    model = model.float().eval().to('cuda')
    directory = FIXTURES / f'{family}-tiny'
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory)
    ids = probe_ids(config.vocab_size)
    with torch.no_grad():
        logits = model(torch.tensor(ids, device='cuda'), use_cache=False).logits.cpu().numpy()
        generated = model.generate(torch.tensor(ids[:1, :4], device='cuda'),
                                   do_sample=False, max_new_tokens=6,
                                   eos_token_id=None, pad_token_id=0).cpu().numpy()
    np.save(directory / 'input_ids.npy', ids)
    np.save(directory / 'logits.npy', logits)
    np.save(directory / 'generated.npy', generated)
    (directory / 'meta.json').write_text(json.dumps({
        'transformers': transformers.__version__, 'torch': torch.__version__,
        'dtype': 'float32', 'attention': 'eager', 'seed': 1234,
        'device': torch.cuda.get_device_name(),
        'command': f'PYTHONPATH=src python tools/classic_gpt_reference.py --fixture --family {family}',
    }, indent=2) + '\n')


def greedy(loaded, ids, steps):
    """Use the decoder cache rather than recomputing the prompt at each draw."""
    model = loaded.model
    variables = loaded.variables
    _, cache = model.apply(variables, jnp.asarray(ids), decode=True, mutable=['cache'])
    logits, cache = model.apply({**variables, **cache}, jnp.asarray(ids),
                                decode=True, mutable=['cache'])
    generated = ids
    for index in range(steps):
        token = np.asarray(jnp.argmax(logits[:, -1], axis=-1), np.int32)[:, None]
        generated = np.concatenate((generated, token), axis=1)
        if index + 1 < steps:
            logits, cache = model.apply({**variables, **cache}, jnp.asarray(token),
                                        decode=True, mutable=['cache'])
    return generated


def check_checkpoint(checkpoint, output, revision=None, safetensors_directory=None):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = AutoConfig.from_pretrained(checkpoint, revision=revision)
    device = 'cuda' if jax.default_backend() == 'gpu' else 'cpu'
    reference = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float32,
                                                    attn_implementation='eager', revision=revision).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision)
    source = checkpoint
    if safetensors_directory is not None:
        # OPT-125m publishes pickle weights alone. Transformers converts
        # the same tensors once; Dew's loader still reads safe tensor storage.
        reference.save_pretrained(safetensors_directory)
        tokenizer.save_pretrained(safetensors_directory)
        source = safetensors_directory
    ids = tokenizer('The capital of France is', return_tensors='np')['input_ids'].astype(np.int32)
    with torch.no_grad():
        expected = reference(torch.tensor(ids, device=device), use_cache=False).logits.cpu().numpy()
        generated = reference.generate(torch.tensor(ids, device=device), do_sample=False,
                                       max_new_tokens=6, eos_token_id=None,
                                       pad_token_id=0).cpu().numpy()
    del reference
    torch.cuda.empty_cache()
    loaded = Pretrained.load(source, dtype='float32', attention_impl='reference', max_seq_len=64,
                             revision=revision)
    loaded = dataclasses.replace(loaded, model=loaded.model.clone(precision=jax.lax.Precision.HIGHEST))
    actual = np.asarray(loaded.model.apply(loaded.variables, jnp.asarray(ids)))
    error = float(np.max(np.abs(expected - actual)))
    # Twice the largest tiny-family error per layer per logit, in fp32 ulps,
    # matches Dew's existing verified-mapping bound (verify._ROUNDING).
    layers = config.num_hidden_layers
    bound = float(2 * _ROUNDING * np.finfo(np.float32).eps * layers * np.max(np.abs(expected)))
    agreement = bool(np.array_equal(actual.argmax(-1), expected.argmax(-1)))
    ours = greedy(loaded, ids, 6)
    metadata = Path(checkpoint) / '.cache' / 'huggingface' / 'download' / 'config.json.metadata'
    revision = config._commit_hash or revision or (metadata.read_text().splitlines()[0] if metadata.is_file() else None)
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
    parser.add_argument('--revision')
    parser.add_argument('--safetensors-directory')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.fixture:
        write_fixture(args.family)
    if args.checkpoint:
        check_checkpoint(args.checkpoint, args.output, args.revision, args.safetensors_directory)


if __name__ == '__main__':
    main()
