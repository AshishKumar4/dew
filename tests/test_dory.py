"""Dory, NVIDIA's looped nGPT, against vLLM's forward (vllm-project/vllm#60149, dory.py at
55076950): tools/dory_reference.py transcribes it and writes tests/fixtures/hf/dory-tiny."""

import dataclasses
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from interop_support import assert_reloads, assert_same_stored_tensors
from reference_error import assert_as_exact_as_the_reference

from dew.interop import Pretrained
from dew.interop.sources import load_shards
from dew.nn.backbones.causal_transformer import CausalTransformer

TINY = Path(__file__).parent / 'fixtures' / 'hf' / 'dory-tiny'
REFERENCE = np.load(TINY / 'reference.npz')


def load():
    return Pretrained.load(TINY, dtype='float32', attention_impl='xla')


@pytest.mark.parametrize('loops', [1, 2, 4])
def test_the_looped_stack_computes_vllms_logits_at_each_loop_count(loops):
    """The tiny Dory's logits with its recurrent block run once, twice (its own count) and four
    times, as vLLM's forward computes them, within tests/reference_error.py's rule of its fp32 run
    against float64."""
    loaded = load()
    model = loaded.model.with_passes(loops)
    logits = model.apply(loaded.variables, REFERENCE['input_ids'])
    assert_as_exact_as_the_reference(np.asarray(logits), REFERENCE[f'logits_{loops}'],
                                     REFERENCE[f'logits_{loops}_f64'], f'Dory logits at {loops} loops')


@pytest.mark.parametrize('mode', ['per_loop', 'last_loop'])
def test_a_cached_greedy_continuation_reads_each_cache_as_vllm_keeps_it(mode):
    """Greedy decoding over the cache vLLM keeps: one entry a recurrent layer and pass
    ('per_loop'), or one a recurrent layer that each pass overwrites ('last_loop', a pass reading
    its own keys for the step's tokens and the last pass's for the earlier ones). The tokens are
    the reference's and each step's logits within the rule of its fp32 decode against float64."""
    loaded = load()
    model = loaded.model.clone(loop=dataclasses.replace(loaded.model.loop, shared_cache=mode == 'last_loop'))
    expected = REFERENCE[f'generated_{mode}']
    prompt = jnp.asarray(expected[None, :6])
    variables = loaded.variables
    cache = model.apply(variables, 1, method=CausalTransformer.init_cache, mutable=['cache'])[1]['cache']
    logits, mutated = model.apply({**variables, 'cache': cache}, prompt, decode=True, mutable=['cache'])
    steps = [np.asarray(logits[0, -1])]
    for token in expected[6:-1]:
        logits, mutated = model.apply({**variables, 'cache': mutated['cache']}, jnp.asarray([[token]]),
                                      decode=True, mutable=['cache'])
        steps.append(np.asarray(logits[0, -1]))
    steps = np.stack(steps)
    np.testing.assert_array_equal(steps.argmax(-1), expected[6:])
    assert_as_exact_as_the_reference(steps, REFERENCE[f'step_logits_{mode}'],
                                     REFERENCE[f'step_logits_{mode}_f64'], f"Dory's {mode} decoding")


def test_a_dory_config_loads_as_its_looped_ngpt():
    """The released config's fields land where Dory computes them: its pairs as blocks, the loop
    over the recurrent ones, reinjecting, and the cache mode."""
    loaded = load()
    model = loaded.model
    assert model.num_layers == 4 and model.ngpt is not None and not model.pre_norms
    assert model.loop is not None and model.loop.layers == (1, 3) and model.loop.steps == 2
    assert model.loop.reinject and not model.loop.shared_cache
    assert model.per_layer_types == ('sliding_attention', 'full_attention') * 2


@pytest.mark.parametrize('change', [{'tie_word_embeddings': True}, {'mlp_bias': True}, {'n_input_layers': 1},
                                    {'recurrent_kv_cache_mode': 'each'}])
def test_a_config_dory_refuses_is_refused(tmp_path, change):
    """What DoryConfig refuses (configs/dory.py:39-133), and a partition that splits an attention
    from its MLP, which Dew's blocks keep together, are refused."""
    for source in TINY.iterdir():
        (tmp_path / source.name).symlink_to(source)
    (tmp_path / 'config.json').unlink()
    stated = json.loads((TINY / 'config.json').read_text())
    (tmp_path / 'config.json').write_text(json.dumps({**stated, **change}))
    with pytest.raises(ValueError):
        Pretrained.load(tmp_path, dtype='float32', attention_impl='xla')


def test_a_saved_checkpoint_reloads_bit_for_bit(tmp_path):
    """A loaded Dory saves its config and tensors as the released checkpoint names them, and the
    saved checkpoint rebuilds the same model and tensors."""
    loaded = Pretrained.load(TINY, dtype='float32', attention_impl='reference')
    loaded.save(tmp_path)
    assert_same_stored_tensors(load_shards(tmp_path), load_shards(TINY))
    assert json.loads((tmp_path / 'config.json').read_text())['model_type'] == 'dory'
    assert_reloads(loaded, tmp_path, REFERENCE['input_ids'])
