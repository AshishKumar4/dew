#!/usr/bin/env python3
"""The Qwen4-Exp fixtures tests/test_qwen4_exp.py and tests/test_hf_reference_decoders.py check against, from
transformers' own Qwen4-Exp (models/qwen4_exp, transformers 5.16.1), on the environment's transformers:

    PYTHONPATH=src:tools .venv-3.12/bin/python tools/qwen4_exp_reference.py

tests/fixtures/hf/qwen4-exp-tiny/ is a random Qwen4ExpForCausalLM of four layers over a width of 64: three
gated delta nets with the release's sigmoid output gate, then a QSA attention layer (four query heads of 32
over two key/value heads, a quarter of each turned, its indexer's sixteen heads of 16 pooling blocks of two
keys and keeping the best two, so the selection parts from dense attention from six visible keys on), every
layer's MLP four routed experts two a token beside a gated shared one, the residual four streams under the
gated read and write with a rank of 8, and the n-gram PLE at layer 2 (one-indexed) hashing bigrams and
trigrams over two heads each; with fp32 logits, left-padded logits and the greedy continuation, and the
transformers class run in the test as the reference. qwen3.8-flash-next/ pins the release's config
(Qwen/Qwen3.8-Flash-Next).

A block scores zero where every indexer head's agreement is negative, and the blocks a tie of zeros picks
among are an order torch's `topk` leaves unspecified, which Dew's (`top_k_selection`) breaks by the lower
index: at two heads a block scores zero one time in four, which on drawn ids decided a query's selection
in three draws of twelve; at sixteen, once in 65536.

The release hashes into primes past 2**24 (20000003 on), which the tiny's table cannot hold, so
release_ple_ids.npz there records the release's own n-gram ids, from transformers' NGramEmbedding at the
release's config with its table left out, for rows that carry end-of-sequence ids inside them and before
them, which a hash stops at.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from hf_reference import write_released_config
from reference_source import padded_probe
from transformers.models.qwen4_exp import modeling_qwen4_exp
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM

from dew.interop.verify import probe_ids, scatter_weights

FIXTURES = Path(__file__).resolve().parents[1] / 'tests' / 'fixtures' / 'hf'
RELEASE = ('Qwen/Qwen3.8-Flash-Next', 'de4b8e4d43b917e7706784d8bb445c9af86a3540')
TINY = {
    'vocab_size': 256, 'hidden_size': 64, 'num_hidden_layers': 4,
    'layer_types': ['linear_attention', 'linear_attention', 'linear_attention', 'full_attention'],
    'num_attention_heads': 4, 'num_key_value_heads': 2, 'head_dim': 32,
    'rope_parameters': {'rope_type': 'default', 'rope_theta': 10000.0, 'partial_rotary_factor': 0.25,
                        'mrope_section': [2, 1, 1], 'mrope_interleaved': True},
    'linear_num_key_heads': 2, 'linear_num_value_heads': 4, 'linear_key_head_dim': 16,
    'linear_value_head_dim': 16, 'linear_conv_kernel_dim': 4, 'output_gate_type': 'sigmoid',
    'num_experts': 4, 'num_experts_per_tok': 2, 'moe_intermediate_size': 32,
    'shared_expert_intermediate_size': 32, 'hc_count': 4, 'hc_lowrank': 8, 'ple_layer_ids': [2],
    'ple_embed_dim': 64, 'ple_conv_kernel_size': 4, 'ngram_size': 3, 'heads_per_ngram': 2,
    'ngram_vocab_size_base': 200, 'make_ngram_vocab_size_divisible_by': 8,
    'split_ngram_parts': 4, 'indexer_n_heads': 16, 'indexer_kv_heads': 1, 'indexer_head_dim': 16,
    'indexer_budget': 4, 'indexer_compress_ratio': 2, 'max_position_embeddings': 128, 'rms_norm_eps': 1e-6,
    'eos_token_id': 2, 'bos_token_id': 2, 'pad_token_id': 0, 'tie_word_embeddings': False}
"""qwen4-exp-tiny's Qwen4ExpTextConfig."""


def logits(model, ids, mask=None, positions=None):
    inputs = {'input_ids': torch.from_numpy(ids).long(), 'use_cache': False}
    if mask is not None:
        inputs.update(attention_mask=torch.from_numpy(mask).long(),
                      position_ids=torch.from_numpy(positions).long())
    with torch.no_grad():
        return model(**inputs).logits.numpy()


def greedy(model, ids, tokens: int):
    """`ids` continued `tokens` tokens by the argmax of a full forward each."""
    for _ in range(tokens):
        ids = np.concatenate([ids, logits(model, ids)[:, -1].argmax(-1)[:, None].astype(ids.dtype)], axis=1)
    return ids


def write_tiny() -> None:
    directory = FIXTURES / 'qwen4-exp-tiny'
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = Qwen4ExpForCausalLM(Qwen4ExpTextConfig(**TINY)).float().eval()
    scatter_weights(model, 1234)
    model.save_pretrained(directory, safe_serialization=True)
    ids = probe_ids(TINY['vocab_size'])
    mask, padded, positions = padded_probe(ids)
    arrays = {'input_ids': ids, 'attention_mask': mask, 'logits': logits(model, ids),
              'padded_logits': logits(model, padded, mask, positions),
              'generated': greedy(model, ids[:, :4], 6)}
    for name, array in arrays.items():
        np.save(directory / f'{name}.npy', array)
    print(f"{directory}: {sorted(path.name for path in directory.iterdir())}")


class _Unallocated(torch.nn.Module):
    """The n-gram table's stand-in: the ids it would gather, each as a row of one, which the module
    flattens back to the ids (its `weight` only names the device the ids go to)."""

    def __init__(self, *_):
        super().__init__()
        self.weight = torch.zeros(())

    def forward(self, ids):
        return ids[..., None]


def release_ple_ids(config: dict) -> dict[str, np.ndarray]:
    """The release's n-gram ids for rows with end-of-sequence ids inside and before them, from
    transformers' `Qwen4ExpTextNGramEmbedding` at the release's text config, its table left out: the
    module hashes, then gathers from a 51B-row table, whose stand-in hands the ids back."""
    text = Qwen4ExpTextConfig(**{key: value for key, value in config['text_config'].items()
                                 if key not in ('model_type', 'architectures')})
    embedding = modeling_qwen4_exp.nn.Embedding
    modeling_qwen4_exp.nn.Embedding = _Unallocated
    try:
        ngram = modeling_qwen4_exp.Qwen4ExpTextNGramEmbedding(text, text.ple_embed_dim, layer_idx=1)
    finally:
        modeling_qwen4_exp.nn.Embedding = embedding
    eos = text.eos_token_id[0] if isinstance(text.eos_token_id, list) else text.eos_token_id
    rows = np.random.RandomState(11).randint(0, text.vocab_size, (3, 16)).astype(np.int64)
    rows[0, 5] = rows[1, 0] = rows[1, 9] = rows[2, 14] = eos
    with torch.no_grad():
        ids = ngram(torch.from_numpy(rows), None).numpy()
    return {'input_ids': rows.astype(np.int32), 'ple_ids': ids.astype(np.int64),
            'layer_multipliers': ngram.layer_multipliers.numpy(),
            'vocab_sizes': ngram.ngram_heads_vocab_sizes.numpy()}


def write_release() -> None:
    directory = FIXTURES / 'qwen3.8-flash-next'
    write_released_config(directory.name, *RELEASE)
    config = json.loads((directory / 'config.json').read_text())
    np.savez(directory / 'release_ple_ids.npz', **release_ple_ids(config))
    print(f"{directory}: {sorted(path.name for path in directory.iterdir())}")


if __name__ == '__main__':
    if sys.argv[1:] == ['--release']:
        write_release()
    else:
        write_tiny()
        write_release()
