"""MiMo-V2-Flash's sliding layers with doubled key and value heads and sinks,
its narrower scaled values and its grouped sigmoid routing, against
transformers 5.16.1: tools/hf_reference.py writes tests/fixtures/hf/mimo-v2-flash-tiny."""

import json
from pathlib import Path

import numpy as np
from reference_error import assert_as_exact_as_the_reference

from dew.interop import Pretrained
from dew.interop.sources import load_shards

ROOT = Path(__file__).parent / 'fixtures' / 'hf'
TINY = ROOT / 'mimo-v2-flash-tiny'


def load():
    return Pretrained.load(TINY, dtype='float32', attention_impl='reference')


def test_padded_logits_and_cached_generation_match_transformers():
    from tools.classic_gpt_reference import greedy

    loaded = load()
    ids = np.load(TINY / 'padded_ids.npy')
    mask = np.load(TINY / 'attention_mask.npy')
    actual = np.asarray(loaded.model.apply(loaded.variables, ids, attention_mask=mask))
    expected, truth = (np.load(TINY / f'padded_logits{suffix}.npy') for suffix in ('', '_f64'))
    assert_as_exact_as_the_reference(actual[mask], expected[mask], truth[mask], 'MiMo-V2-Flash padded logits')
    np.testing.assert_array_equal(actual[mask].argmax(-1), expected[mask].argmax(-1))
    original = np.load(TINY / 'input_ids.npy')
    np.testing.assert_array_equal(greedy(loaded, original[:, :4], 6), np.load(TINY / 'generated.npy'))


def test_export_reads_in_transformers_and_reloads_bit_for_bit(tmp_path):
    import torch
    from transformers import AutoModelForCausalLM

    loaded = load()
    loaded.save(tmp_path)
    assert json.loads((tmp_path / 'config.json').read_text())['model_type'] == 'mimo_v2_flash'
    original, exported = load_shards(TINY), load_shards(tmp_path)
    assert set(original) == set(exported)
    for tensor in original:
        np.testing.assert_array_equal(original[tensor], exported[tensor])
    reference = AutoModelForCausalLM.from_pretrained(tmp_path, attn_implementation='eager').float().eval()
    reference.set_experts_implementation('eager')
    ids = np.load(TINY / 'input_ids.npy')
    with torch.no_grad():
        expected = reference(torch.from_numpy(ids).long(), use_cache=False).logits.numpy()
    actual = loaded.model.apply(loaded.variables, ids)
    truth = np.load(TINY / 'logits_f64.npy')
    assert_as_exact_as_the_reference(actual, expected, truth, 'MiMo-V2-Flash export')
    restored = Pretrained.load(tmp_path, dtype='float32', attention_impl='reference')
    np.testing.assert_array_equal(restored.model.apply(restored.variables, ids), actual)
