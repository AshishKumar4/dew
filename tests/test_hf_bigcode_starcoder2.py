"""GPTBigCode's fused multi-query and multi-head c_attn and Starcoder2's
windowed grouped-query block against transformers 5.16.1."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference

from dew.interop import Pretrained, PretrainedDecoder
from dew.interop.hf_decoders import translate_config
from dew.interop.sources import load_shards

ROOT = Path(__file__).parent / 'fixtures' / 'hf'
FIXTURES = ['gpt-bigcode-tiny', 'gpt-bigcode-mha-tiny', 'starcoder2-tiny']


def load(name):
    return Pretrained.load(ROOT / name, dtype='float32', attention_impl='reference')


@pytest.mark.parametrize('name', FIXTURES)
def test_padded_logits_and_cached_generation_match_transformers(name):
    from tools.classic_gpt_reference import greedy

    directory = ROOT / name
    loaded = load(name)
    ids = np.load(directory / 'padded_ids.npy')
    mask = np.load(directory / 'attention_mask.npy')
    actual = np.asarray(loaded.model.apply(loaded.variables, ids, attention_mask=mask))
    expected, truth = (np.load(directory / f'padded_logits{suffix}.npy') for suffix in ('', '_f64'))
    assert_as_exact_as_the_reference(actual[mask], expected[mask], truth[mask], f'{name} padded logits')
    np.testing.assert_array_equal(actual[mask].argmax(-1), expected[mask].argmax(-1))
    original = np.load(directory / 'input_ids.npy')
    np.testing.assert_array_equal(greedy(loaded, original[:, :4], 6), np.load(directory / 'generated.npy'))


@pytest.mark.parametrize('name, exported_type', [('gpt-bigcode-tiny', 'gpt_bigcode'),
                                                 ('gpt-bigcode-mha-tiny', 'gpt2'),
                                                 ('starcoder2-tiny', 'starcoder2')])
def test_export_reads_in_transformers_and_reloads_bit_for_bit(name, exported_type, tmp_path):
    """A multi-query GPTBigCode and Starcoder2 write their own tensors back
    unchanged; a multi-head GPTBigCode computes GPT-2's model, so it saves
    as GPT-2. Each export reads in transformers to the same logits."""
    import torch
    from transformers import AutoModelForCausalLM

    directory = ROOT / name
    loaded = load(name)
    PretrainedDecoder.from_model(loaded.model, loaded.variables).save(tmp_path)
    assert json.loads((tmp_path / 'config.json').read_text())['model_type'] == exported_type
    if exported_type != 'gpt2':
        original, exported = load_shards(directory), load_shards(tmp_path)
        assert set(original) == set(exported)
        for tensor in original:
            np.testing.assert_array_equal(original[tensor], exported[tensor])
    reference = AutoModelForCausalLM.from_pretrained(tmp_path, attn_implementation='eager').float().eval()
    ids = np.load(directory / 'input_ids.npy')
    with torch.no_grad():
        expected = reference(torch.from_numpy(ids).long(), use_cache=False).logits.numpy()
    actual = loaded.model.apply(loaded.variables, ids)
    truth = np.load(directory / 'logits_f64.npy')
    assert_as_exact_as_the_reference(actual, expected, truth, f'{name} export')
    restored = Pretrained.load(tmp_path, dtype='float32', attention_impl='reference')
    np.testing.assert_array_equal(restored.model.apply(restored.variables, ids), actual)


def test_the_reference_catches_a_multi_head_c_attn_read_as_contiguous_q_k_v():
    """A multi-head c_attn holds each head's q, k and v side by side; read as
    all queries, then all keys, then all values (the multi-query layout), the
    same tensor computes another model."""
    from dew.interop.families.gpt_neox import gpt_neox_prepare

    directory = ROOT / 'gpt-bigcode-mha-tiny'
    loaded = load('gpt-bigcode-mha-tiny')
    model = loaded.model
    config = {'num_heads': model.num_heads, 'head_dim': model.head_dim, 'num_kv_heads': model.num_heads}
    contiguous = gpt_neox_prepare(load_shards(directory), config, attention_name='attn', fused_name='c_attn',
                                  interleaved=False)
    params = jax.tree.map(jnp.asarray, loaded.variables)['params']
    for index in range(model.num_layers):
        for part in ('q_proj', 'k_proj', 'v_proj'):
            stem = f'transformer.h.{index}.self_attn.{part}'
            params[f'layers_{index}']['self_attn'][part] = {
                'kernel': jnp.asarray(contiguous[f'{stem}.weight'].T),
                'bias': jnp.asarray(contiguous[f'{stem}.bias'])}
    wrong = model.apply({'params': params}, np.load(directory / 'input_ids.npy'))
    with pytest.raises(AssertionError, match=r'allowed 2\.0'):
        assert_as_exact_as_the_reference(wrong, np.load(directory / 'logits.npy'),
                                         np.load(directory / 'logits_f64.npy'), 'GPTBigCode contiguous qkv')


def test_the_reference_catches_starcoder2_without_its_window():
    """The fixture's five-key window is shorter than its twelve tokens, so
    attending every key computes another model."""
    directory = ROOT / 'starcoder2-tiny'
    loaded = load('starcoder2-tiny')
    full = loaded.model.clone(layer_types=('full_attention',) * loaded.model.num_layers, kinds={})
    wrong = full.apply(loaded.variables, np.load(directory / 'input_ids.npy'))
    with pytest.raises(AssertionError, match=r'allowed 2\.0'):
        assert_as_exact_as_the_reference(wrong, np.load(directory / 'logits.npy'),
                                         np.load(directory / 'logits_f64.npy'), 'Starcoder2 unwindowed')


def test_a_model_the_starcoder2_config_cannot_state_does_not_export_as_one():
    """Starcoder2's config holds one `use_bias` and one window for every
    layer, so a model biasing its feed-forward alone, or windowing one layer
    of two, is not exported as Starcoder2."""
    from dew.interop.families.starcoder2 import STARCODER2

    model = load('starcoder2-tiny').model
    assert STARCODER2.matches(model)
    assert not STARCODER2.matches(model.clone(attention_bias=False))
    assert not STARCODER2.matches(model.clone(o_proj_bias=False))
    mixed = ('sliding_attention', 'full_attention') * (model.num_layers // 2)
    assert not STARCODER2.matches(model.clone(layer_types=mixed))


def test_the_released_configs_translate_their_computation():
    released = json.loads((ROOT / 'gpt-bigcode-santacoder' / 'config.json').read_text())
    santacoder = translate_config(released).value
    assert (santacoder.emb_features, santacoder.num_heads, santacoder.num_layers) == (2048, 16, 24)
    assert santacoder.kv_heads == 1 and santacoder.position_embedding == 'learned'
    assert santacoder.mlp == 'gelu' and santacoder.norm_type == 'layer' and santacoder.attention_bias
    starcoder2 = translate_config(json.loads((ROOT / 'starcoder2-3b' / 'config.json').read_text())).value
    assert (starcoder2.emb_features, starcoder2.num_heads, starcoder2.kv_heads) == (3072, 24, 2)
    assert starcoder2.per_layer_types == ('sliding_attention',) * 30
    assert starcoder2.kind_of('sliding_attention').window == 4096
    assert starcoder2.rope_theta == pytest.approx(999999.4420358813)
    assert starcoder2.mlp_bias and starcoder2.norm_bias and starcoder2.tie_embeddings


@pytest.mark.network
@pytest.mark.parametrize('name', ['gpt-bigcode-santacoder', 'starcoder2-3b'])
def test_the_released_configs_are_pinned(name):
    from huggingface_hub import hf_hub_download

    source = json.loads((ROOT / name / 'source.json').read_text())
    filename = hf_hub_download(source['repo'], 'config.json', revision=source['revision'])
    assert json.loads(Path(filename).read_text()) == json.loads((ROOT / name / 'config.json').read_text())
