"""Phi-3's fused GQA and LongRoPE against transformers 5.16.1."""

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
from dew.nn.rope import LongRopeScaling

ROOT = Path(__file__).parent / 'fixtures' / 'hf'
DIRECTORY = ROOT / 'phi3-tiny'


@pytest.mark.parametrize('padded', [False, True])
def test_phi3_short_and_long_logits_match_transformers(padded):
    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    ids = np.load(DIRECTORY / ('padded_ids.npy' if padded else 'input_ids.npy'))
    mask = np.load(DIRECTORY / 'attention_mask.npy') if padded else np.ones_like(ids, bool)
    prefix = 'padded_' if padded else ''
    actual = np.asarray(loaded.model.apply(loaded.variables, ids, attention_mask=mask))
    expected, truth = (np.load(DIRECTORY / f'{prefix}logits{suffix}.npy') for suffix in ('', '_f64'))
    assert_as_exact_as_the_reference(actual[mask], expected[mask], truth[mask], 'Phi-3 long logits')
    np.testing.assert_array_equal(actual[mask].argmax(-1), expected[mask].argmax(-1))
    if not padded:
        short = loaded.model.apply(loaded.variables, ids[:, :4])
        assert_as_exact_as_the_reference(short, np.load(DIRECTORY / 'short_logits.npy'),
                                         np.load(DIRECTORY / 'short_logits_f64.npy'), 'Phi-3 short logits')


def test_phi3_cached_greedy_crosses_the_longrope_boundary():
    from dew.inference.tasks import TextGeneration
    from dew.sampling.text import Sampling

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference', max_seq_len=48)
    ids = np.load(DIRECTORY / 'input_ids.npy')
    task = TextGeneration(loaded.model, loaded.variables, sampling=Sampling(temperature=0))
    generated = task(ids[:, :4], max_new_tokens=6, key=0)
    np.testing.assert_array_equal(generated.tokens, np.load(DIRECTORY / 'generated.npy'))


def test_phi3_cache_rebuild_logits_match_uncached_transformers_on_both_sides_of_the_crossing():
    from dew.nn.inputs import ModelInputs
    from dew.sampling.text import _operations, _prefill

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference', max_seq_len=48)
    generated = np.load(DIRECTORY / 'generated.npy')
    ops = _operations(loaded.model, loaded.variables, 0, 0)
    state, _ = _prefill(loaded.model, loaded.variables,
                        ModelInputs(jnp.asarray(generated[:, :4], jnp.int32)), ops)
    outputs = []
    for step in range(6):
        outputs.append(np.asarray(state.logits))
        state = ops.advance(state, jnp.asarray(generated[:, 4 + step]), jnp.ones(2, bool))
    actual = np.stack(outputs, axis=1)
    reference = np.load(DIRECTORY / 'generation_logits.npy')
    truth = np.load(DIRECTORY / 'generation_logits_f64.npy')
    for step in range(6):
        assert_as_exact_as_the_reference(actual[:, step], reference[:, step], truth[:, step],
                                         f'Phi-3 crossing step {step}')
    np.testing.assert_array_equal(actual.argmax(-1), generated[:, 4:])


def test_phi3_export_preserves_both_fusions_and_reads_in_transformers(tmp_path):
    import torch
    from transformers import AutoModelForCausalLM

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    PretrainedDecoder.from_model(loaded.model, loaded.variables).save(tmp_path)
    original, exported = load_shards(DIRECTORY), load_shards(tmp_path)
    assert set(original) == set(exported)
    for name in original:
        np.testing.assert_array_equal(original[name], exported[name])
    reference = AutoModelForCausalLM.from_pretrained(tmp_path, attn_implementation='eager').float().eval()
    ids = np.load(DIRECTORY / 'input_ids.npy')
    with torch.no_grad():
        expected = reference(torch.from_numpy(ids).long(), use_cache=False).logits.numpy()
    actual = np.asarray(loaded.model.apply(loaded.variables, ids))
    assert_as_exact_as_the_reference(actual, expected, np.load(DIRECTORY / 'logits_f64.npy'), 'Phi-3 export')
    restored = Pretrained.load(tmp_path, dtype='float32', attention_impl='reference')
    np.testing.assert_array_equal(restored.model.apply(restored.variables, ids), actual)


def test_phi3_reference_catches_the_short_table_used_for_a_long_call():
    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    rope = loaded.model.rope_scaling
    assert isinstance(rope, LongRopeScaling)
    wrong_rope = LongRopeScaling(rope.short_factor, rope.short_factor,
                                rope.original_max_position_embeddings, rope.factor, rope.attention_factor)
    wrong = loaded.model.clone(rope_scaling=wrong_rope).apply(
        loaded.variables, np.load(DIRECTORY / 'input_ids.npy'))
    with pytest.raises(AssertionError, match=r'allowed 2\.0'):
        assert_as_exact_as_the_reference(wrong, np.load(DIRECTORY / 'logits.npy'),
                                         np.load(DIRECTORY / 'logits_f64.npy'), 'Phi-3 wrong table')


def test_phi3_gated_block_has_finite_gradients_and_updates():
    import optax

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    ids = jnp.asarray(np.load(DIRECTORY / 'input_ids.npy'))

    def loss(params):
        return optax.softmax_cross_entropy_with_integer_labels(
            loaded.model.apply({'params': params}, ids[:, :-1]), ids[:, 1:]).mean()

    value, gradient = jax.value_and_grad(loss)(loaded.variables['params'])
    changed = jax.tree.map(lambda parameter, grad: parameter - 1e-3 * grad,
                           loaded.variables['params'], gradient)
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(gradient))
    assert float(loss(changed)) < float(value)


@pytest.mark.parametrize('name', ['phi3-mini-4k', 'phi4-mini'])
def test_phi3_released_configs_build_the_native_fused_geometry(name):
    model = translate_config(json.loads((ROOT / name / 'config.json').read_text())).value
    assert model.emb_features == 3072
    assert model.qk_norm is False and model.mlp == 'swiglu'
    if name == 'phi4-mini':
        assert model.num_heads == 24 and model.features_per_head == 128
        assert model.partial_rotary_factor == .75 and model.kv_heads == 8
        assert isinstance(model.rope_scaling, LongRopeScaling)
        assert len(model.rope_scaling.short_factor) == 48
    else:
        assert model.num_heads == 32 and model.features_per_head == 96
        assert model.rope_scaling is None and model.kind_of('sliding_attention').window == 2047


@pytest.mark.network
@pytest.mark.parametrize('name', ['phi3-mini-4k', 'phi4-mini'])
def test_phi3_released_configs_are_pinned(name):
    from huggingface_hub import hf_hub_download

    source = json.loads((ROOT / name / 'source.json').read_text())
    filename = hf_hub_download(source['repo'], 'config.json', revision=source['revision'])
    assert json.loads(Path(filename).read_text()) == json.loads((ROOT / name / 'config.json').read_text())
