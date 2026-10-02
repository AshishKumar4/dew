"""Measure GPT-NeoX layer errors and fp32 rounding against float64."""

import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dew.interop import load_pretrained
from dew.nn.backbones.causal_transformer import layer_output, layer_outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    reference = AutoModelForCausalLM.from_pretrained(args.checkpoint, dtype=torch.float32,
                                                    attn_implementation='eager').eval().to('cuda')
    ids = AutoTokenizer.from_pretrained(args.checkpoint)('The capital of France is', return_tensors='np')['input_ids']
    with torch.no_grad():
        expected = reference(torch.tensor(ids, device='cuda'), use_cache=False, output_hidden_states=True)
    loaded = load_pretrained(args.checkpoint, dtype='float32', attention_impl='reference', max_seq_len=64)
    model = loaded.model.clone(precision=jax.lax.Precision.HIGHEST)
    actual, intermediates = model.apply(loaded.variables, jnp.asarray(ids),
                                        capture_intermediates=layer_outputs, mutable=['intermediates'])
    observations = []
    for index in range(model.num_layers - 1):
        mine = np.asarray(layer_output(intermediates['intermediates'], index))
        theirs = expected.hidden_states[index + 1].cpu().numpy()
        observations.append({'layer': index, 'max_error': float(np.max(np.abs(mine - theirs))),
                             'max_hidden': float(np.max(np.abs(theirs)))})
    parts = {}
    index = 3
    given = expected.hidden_states[index]
    layer = model.bind(loaded.variables).layers[index]
    ref_layer = reference.gpt_neox.layers[index]
    for name in ('input_layernorm', 'post_attention_layernorm'):
        with torch.no_grad():
            ref_norm = getattr(ref_layer, name)(given).cpu().numpy()
        ours = np.asarray(getattr(layer, name)(jnp.asarray(given.cpu().numpy())))
        parts[name] = {'max_error': float(np.max(np.abs(ours - ref_norm))),
                       'max_reference': float(np.max(np.abs(ref_norm)))}
        if name == 'post_attention_layernorm':
            with torch.no_grad():
                ff = ref_layer.mlp(torch.tensor(ref_norm, device='cuda')).cpu().numpy()
            my_ff = np.asarray(layer.mlp(jnp.asarray(ref_norm)))
            parts['mlp_same_normed_input'] = {'max_error': float(np.max(np.abs(my_ff - ff))),
                                             'max_reference': float(np.max(np.abs(ff)))}
    with torch.no_grad():
        truth = reference.double()(torch.tensor(ids, device='cuda'), use_cache=False).logits.cpu().numpy()
    theirs = expected.logits.cpu().numpy()
    mine = np.asarray(actual)
    distance = lambda left, right: float(np.sqrt(np.mean((np.asarray(left, np.float64) - np.asarray(right, np.float64)) ** 2)))
    print(json.dumps({'layers': observations, 'layer3_components': parts,
                      'max_error': float(np.max(np.abs(mine - theirs))),
                      'max_reference_logit': float(np.max(np.abs(theirs))),
                      'reference_fp32_vs_fp64_rms': distance(theirs, truth),
                      'dew_fp32_vs_reference_fp64_rms': distance(mine, truth)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
