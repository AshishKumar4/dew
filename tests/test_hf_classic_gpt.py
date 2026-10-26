"""Classic decoder ports against transformers 5.16.1, with nonzero biases."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.batch import coordinates
from dew.interop import Pretrained
from dew.interop.hf_decoders import translate_config
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.sampling import Sampling, generate

FIXTURES = Path(__file__).parent / 'fixtures' / 'hf'


def test_gpt2_same_weight_logits_and_cached_generation():
    """The biased two-projection block matches fp32 eager transformers.

    Observed max error 2.38e-7 on CPU against the RTX 4080 reference at
    highest precision. The 1e-4 bound is the existing dense-family fp32 bound. Scattered
    LayerNorm scales, biases and learned positions make dropping any of
    these paths a numerical failure.
    """
    directory = FIXTURES / 'gpt2-tiny'
    loaded = Pretrained.load(directory, dtype='float32', attention_impl='reference')
    model = loaded.model.clone(precision=jax.lax.Precision.HIGHEST)
    ids = np.load(directory / 'input_ids.npy')
    expected = np.load(directory / 'logits.npy')
    actual = np.asarray(model.apply(loaded.variables, jnp.asarray(ids)))
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=0)
    np.testing.assert_array_equal(actual.argmax(-1), expected.argmax(-1))
    generated = generate(model, loaded.variables, ids[:1, :4], 6, key=0, sampling=Sampling(temperature=0))
    np.testing.assert_array_equal(np.asarray(generated.tokens), np.load(directory / 'generated.npy'))


def test_gpt2_public_generation_matches_the_reference_continuation():
    from dew.inference.tasks import TextGeneration
    from dew.sampling.text import Sampling

    loaded = Pretrained.load(FIXTURES / 'gpt2-tiny', dtype='float32', attention_impl='reference')
    ids = np.load(FIXTURES / 'gpt2-tiny' / 'input_ids.npy')[:1, :4]
    task = TextGeneration(loaded.model, loaded.variables, sampling=Sampling(temperature=0))
    generated = task(ids, max_new_tokens=6, key=0)
    np.testing.assert_array_equal(
        np.asarray(generated.tokens), np.load(FIXTURES / "gpt2-tiny" / "generated.npy")
    )


def test_original_bare_gpt2_weights_and_causal_buffers_preserve_logits():
    from dew.interop.hf_decoders import translate_weights
    from dew.interop.sources import load_shards

    directory = FIXTURES / 'gpt2-tiny'
    loaded = Pretrained.load(directory, dtype='float32', attention_impl='reference')
    tensors = {name.removeprefix('transformer.'): value for name, value in load_shards(directory).items()}
    for index in range(loaded.model.num_layers):
        tensors[f'h.{index}.attn.bias'] = np.tril(np.ones((1, 1, 64, 64), np.uint8))
        tensors[f'h.{index}.attn.masked_bias'] = np.asarray(-1e4, np.float32)
    config = translate_config(loaded.config)
    variables = translate_weights(tensors, config, 'gpt2')
    ids = jnp.asarray(np.load(directory / 'input_ids.npy'))
    np.testing.assert_array_equal(
        loaded.model.apply(variables, ids), loaded.model.apply(loaded.variables, ids)
    )
    tensors['h.0.attn.bias'][0, 0, 0, 1] = 1
    with pytest.raises(ValueError):
        translate_weights(tensors, config, 'gpt2')


@pytest.mark.network
def test_released_gpt2_small_logits_and_generation_at_highest_fp32_precision(tmp_path):
    from tools.classic_gpt_reference import check_checkpoint

    check_checkpoint('openai-community/gpt2', tmp_path / 'parity.json',
                     revision='607a30d783dfa663caf39e06633721c8d4cfcd7e')


@pytest.mark.parametrize('bias', [False, True])
def test_exact_nanogpt_architecture_trains_through_dew(bias):
    """An ungated exact-GELU GPT block goes through the public trainer."""
    import optax

    from dew import Trainer
    from dew.data.dataset import Loader
    from dew.objectives.lm import LMObjective

    model = CausalTransformer(vocab_size=32, emb_features=16, num_layers=2, num_heads=2,
                              max_seq_len=8, position_embedding='learned', mlp='gelu_exact',
                              mlp_bias=bias, norm_type='layer', norm_bias=bias,
                              attention_bias=bias, qk_norm=False, attention_impl='reference',
                              dropout_rate=.2, embedding_dropout_rate=.2, attention_dropout_rate=.2)
    rows = np.asarray([[2, 4, 8, 3, 2, 4, 8, 3, 2], [7, 3, 9, 5, 7, 3, 9, 5, 7]], np.int32)
    rows = np.tile(rows, (jax.device_count(), 1))
    variables = model.init(jax.random.key(0), jnp.asarray(rows[:, :-1]))
    objective = LMObjective(model, seq_len=8, ema_decay=None, variables=variables)
    data = Loader(train=lambda partition: iter({'text': rows} for _ in range(3)),
                   val=None, records=3 * len(rows), batch=len(rows))
    trainer = Trainer(objective, optax.adam(1e-3), key=jax.random.key(0))
    before = np.asarray(model.apply(variables, jnp.asarray(rows[:, :-1])))
    state = trainer.fit(data, steps=3)
    after = np.asarray(model.apply(state.variables, jnp.asarray(rows[:, :-1])))
    def loss(logits):
        return float(optax.softmax_cross_entropy_with_integer_labels(logits, jnp.asarray(rows[:, 1:])).mean())
    assert np.isfinite(loss(after)) and loss(after) < loss(before)


@pytest.mark.parametrize('given', ['attention_mask', 'coordinates'])
@pytest.mark.parametrize('scan_layers', [False, True])
def test_learned_position_cache_advances_each_rows_unpadded_positions(scan_layers, given):
    """Learned positions read each row's unpadded places, whether the padding comes as a raw mask
    or as the tokens' coordinates alone, prefill and cache alike."""
    model = CausalTransformer(vocab_size=16, emb_features=8, num_layers=2, num_heads=2,
                              max_seq_len=8, position_embedding='learned', qk_norm=False,
                              attention_impl='reference', scan_layers=scan_layers)
    ids = jnp.asarray([[1, 2, 3, 4], [0, 0, 5, 6]], jnp.int32)
    valid = jnp.asarray([[True, True, True, True], [False, False, True, True]])
    variables = model.init(jax.random.key(0), ids)
    padding = {'attention_mask': valid} if given == 'attention_mask' else {
        'coordinates': coordinates(valid=valid)}
    _, cache = model.apply(variables, ids, decode=True, mutable=['cache'], **padding)
    _, cache = model.apply({**variables, **cache}, ids, decode=True, mutable=['cache'], **padding)
    token = jnp.asarray([[7], [8]], jnp.int32)
    cached, _ = model.apply({**variables, **cache}, token, decode=True, mutable=['cache'])
    for index, prompt in enumerate((ids[0], ids[1, 2:])):
        joined = jnp.concatenate((prompt, token[index]), axis=0)[None]
        full = model.apply(variables, joined)
        np.testing.assert_allclose(cached[index, -1], full[0, -1], atol=1e-5, rtol=0)
