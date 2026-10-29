"""Ouro's looped decoder against its own remote code on transformers 4.56.2:
tools/ouro_reference.py writes tests/fixtures/hf/ouro-tiny and the 1.4B probe."""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from interop_support import assert_reloads, assert_same_stored_tensors, fixture_ids
from reference_error import assert_as_exact, declared_seeds, indexed_recording, recording, widened

from dew.interop import Pretrained, PretrainedDecoder
from dew.interop.sources import load_shards
from dew.nn.backbones.decoder_stack import Loop

ROOT = Path(__file__).parent / 'fixtures' / 'hf'
TINY = ROOT / 'ouro-tiny'
RELEASED = ROOT / 'ouro-1.4b'
SOURCE = 'ByteDance/Ouro-1.4B'


def load():
    return Pretrained.load(TINY, dtype='float32', attention_impl='reference')


def fixture(name):
    return np.load(TINY / f'{name}.npy')



def test_every_pass_gates_its_exit_as_the_remote_code_does():
    """The logits and each pass's exit-gate logits are on par with the remote code's fp32 run from its
    float64 run (check 1), over its recording's seeds of ids (tools/ouro_reference.py `--recording`):
    Dew's own float64 run is the truth, held at every seed to the recording's sketch."""
    loaded = load()
    wide = loaded.model.clone(dtype=jnp.float64)
    path = TINY / 'recorded_logits.npz'
    seeds = declared_seeds(path)['logits']
    draw = fixture_ids(TINY)

    def passes(model, variables):
        def run(ids):
            logits, sown = model.apply(variables, ids, mutable=['exits'])
            return {'logits': logits, 'exit_logits': jnp.stack(sown['exits']['logits'])}
        return run

    truth_of = passes(wide, widened(loaded.variables))
    recorded, reference, truth = indexed_recording(
        recording(path), 'logits', seeds, truth_fn=lambda inputs: truth_of(inputs['input_ids']),
        inputs_of=lambda seed: {'input_ids': draw(seed)[0]}, leaves=('logits', 'exit_logits'))
    ours = passes(loaded.model, loaded.variables)
    assert_as_exact(lambda index, inputs: ours(inputs['input_ids']), reference, recorded, 'Ouro passes',
                    seeds=seeds, truth_fn=truth)


def test_a_saved_checkpoint_reloads_bit_for_bit(tmp_path):
    loaded = load()
    loaded.save(tmp_path)
    assert_same_stored_tensors(load_shards(tmp_path), load_shards(TINY))
    assert_reloads(loaded, tmp_path, fixture('input_ids'))


def test_a_looped_model_exports_in_the_remote_codes_layout(tmp_path):
    """A model built in Dew writes the config Ouro's remote code reads, which
    names that code; a loop it would not compute, one without the norm between
    passes, is refused, while how the loop trained is not part of it."""
    loaded = load()

    def exported(**loop):
        return PretrainedDecoder.from_model(loaded.model.clone(loop=Loop(3, exit_gate=True, **loop)),
                                            loaded.variables)

    exported(backprop_steps=1).save(tmp_path)
    config = json.loads((tmp_path / 'config.json').read_text())
    assert config['model_type'] == 'ouro' and config['total_ut_steps'] == 3
    assert config['early_exit_threshold'] == 1.0
    assert config['auto_map']['AutoModelForCausalLM'] == f'{SOURCE}--modeling_ouro.OuroForCausalLM'
    assert set(load_shards(tmp_path)) == set(load_shards(TINY))
    ids = fixture('input_ids')
    restored = Pretrained.load(tmp_path, dtype='float32', attention_impl='reference')
    np.testing.assert_array_equal(restored.model.apply(restored.variables, ids),
                                  loaded.model.apply(loaded.variables, ids))
    with pytest.raises(ValueError):
        exported(step_norm=False)


@pytest.mark.network
def test_ouro_1_4b_computes_the_remote_codes_logits():
    """ByteDance/Ouro-1.4B in fp32 on CPU, against the remote code's argmax
    and 64 of its logit columns on the probe ids (probe.npz)."""
    source = json.loads((RELEASED / 'source.json').read_text())
    probe = np.load(RELEASED / 'probe.npz')
    loaded = Pretrained.load(source['repo'], revision=source['revision'], dtype='float32')
    logits = np.asarray(loaded.model.apply(loaded.variables, probe['input_ids']), np.float32)
    np.testing.assert_array_equal(logits.argmax(-1), probe['argmax'])
    assert float(np.max(np.abs(logits[..., probe['columns']] - probe['logits']))) < 1e-3
