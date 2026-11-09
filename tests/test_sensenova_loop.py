"""SenseNova-Loop's language model against its own remote code (sensenova/SenseNova-Loop-9B-Preview,
modeling_sensenova_loop.py): tools/sensenova_loop_reference.py writes tests/fixtures/hf/sensenova-loop-tiny
and its recording."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from reference_error import assert_as_exact, declared_seeds, indexed_recording, recording, widened

from dew.interop import Pretrained
from dew.nn.backbones.decoder_stack import Loop
from dew.nn.hyper_connections import HyperConnections

TINY = Path(__file__).parent / 'fixtures' / 'hf' / 'sensenova-loop-tiny'
SHAPE = (4, 48)
"""The ids' shape check 1 draws in, as the recorder drew them."""


def load():
    return Pretrained.load(TINY, text_only=True, dtype='float32', attention_impl='reference')


def draw(seed):
    """Check 1's ids, uniform over the language model's vocabulary, as the recorder drew them."""
    vocabulary = json.loads((TINY / 'config.json').read_text())['text_config']['vocab_size']
    return np.random.default_rng(seed).integers(0, vocabulary, SHAPE).astype(np.int32)


def test_its_language_model_loops_a_block_under_streams_over_it():
    """The release's loop and mHC settings, stated at its wrapper's scope, read as its language model's:
    the block [2, 6) run twice, each pass reading the last one's output as it is, and two streams over the
    block under the release's routers' settings (`families.sensenova`)."""
    model = load().model
    assert model.loop == Loop(2, step_norm=False, layers=(2, 6))
    assert model.hyper_connections == HyperConnections(
        hc_mult=2, hc_eps=0.0, hc_sinkhorn_iters=20, head='mean', hc_scaled=False, hc_mix_transposed=True,
        hc_pre_scale=0.75, hc_layers=(2, 6))


def test_its_logits_are_on_par_with_the_remote_codes():
    """The text-only load's logits on par with the remote code's fp32 run from its float64 run (check 1),
    over its recording's seeds of ids (tools/sensenova_loop_reference.py `--recording`): Dew's own float64
    run is the truth, held at every seed to the recording's sketch."""
    loaded = load()
    wide = loaded.model.clone(dtype=jnp.float64)
    path = TINY / 'recorded_logits.npz'
    seeds = declared_seeds(path)['logits']
    truth_of = jax.jit(lambda ids: {'logits': wide.apply(widened(loaded.variables), ids)})
    recorded, reference, truth = indexed_recording(
        recording(path), 'logits', seeds, truth_fn=lambda inputs: truth_of(inputs['input_ids']),
        inputs_of=lambda seed: {'input_ids': draw(seed)}, leaves=('logits',))
    ours = jax.jit(lambda ids: {'logits': loaded.model.apply(loaded.variables, ids)})
    assert_as_exact(lambda index, inputs: ours(inputs['input_ids']), reference, recorded, 'SenseNova-Loop',
                    seeds=seeds, truth_fn=truth)
