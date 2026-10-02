"""Python training saves the model/task declaration with its own checkpoint."""
import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew.checkpoints import Checkpoints
from dew.config import ModelConfig
from dew.data import Dataset, Loading
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.lm import LMObjective
from dew.training import Trainer


def model():
    return CausalTransformer(vocab_size=16, emb_features=16, num_layers=1, num_heads=2,
                             mlp_features=32, max_seq_len=16, dtype='float32', attention_impl='xla')


def test_live_model_record_rebuilds_exact_constructor_fields():
    original = model()
    rebuilt = ModelConfig.from_model(original).build()
    assert rebuilt == original.clone(dtype=jnp.float32)
    tokens = jnp.arange(1, 9)[None, :]
    variables = original.init(jax.random.key(0), tokens)
    np.testing.assert_array_equal(original.apply(variables, tokens), rebuilt.apply(variables, tokens))


def test_python_lm_run_saves_its_inference_record_without_run_json(tmp_path):
    objective = LMObjective(model(), seq_len=8, ema_decay=None)
    rows = [{'text': np.arange(9, dtype=np.int32)} for _ in range(16)]
    data = Dataset.from_grain(grain.MapDataset.source(rows), batch=8, loading=Loading(workers=0))
    checkpoints = Checkpoints(str(tmp_path / 'run'))
    trainer = Trainer(objective, optax.sgd(.01), key=0, checkpoints=checkpoints)
    state = trainer.fit(data, steps=2, log_every=2, checkpoint_every=1)
    checkpoints.wait()
    assert not (tmp_path / 'run' / 'run.json').exists()
    record = Checkpoints(str(tmp_path / 'run')).artifact(2)
    assert record['objective'] == 'lm'
    assert record['seq_len'] == 8
    from dew.interop import Pretrained, PretrainedDecoder, PretrainedMaskedDecoder
    bundle = Pretrained.from_run(tmp_path / 'run')
    assert isinstance(bundle, PretrainedDecoder)
    with pytest.raises(TypeError, match='PretrainedDecoder source, not a PretrainedMaskedDecoder'):
        PretrainedMaskedDecoder.from_run(tmp_path / 'run')
    rebuilt = ModelConfig.from_dict(record['model']).build()
    assert rebuilt == objective.model.clone(dtype=jnp.float32)
    assert record['tokenizer'] is None
    from dew.inference import TextGeneration
    task = TextGeneration.from_run(str(tmp_path / 'run'), ema=False)
    assert task.processor is None
    tokens = jnp.arange(1, 9)[None, :]
    expected = objective.model.apply(state.params, tokens)
    actual = task.model.apply(task.variables, tokens)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_builtin_process_records_preserve_noise_prediction_and_weights():
    from dew.diffusion.presets import EDM, Cosine, Flow
    from dew.diffusion.process import Process
    for preset in (EDM(regime='pixel'), Flow(), Cosine()):
        original = preset()
        rebuilt = Process.from_json(original.to_json())
        time = jnp.linspace(.01, .99, 16)
        np.testing.assert_array_equal(original.schedule.rates(time)[0], rebuilt.schedule.rates(time)[0])
        np.testing.assert_array_equal(original.schedule.rates(time)[1], rebuilt.schedule.rates(time)[1])
        np.testing.assert_array_equal(original.weight(time), rebuilt.weight(time))
        rates = original.schedule.rates(time)
        clean, noise = jnp.ones((16, 1)), jnp.full((16, 1), .2)
        np.testing.assert_array_equal(original.prediction.get_target(clean, noise, rates),
                                      rebuilt.prediction.get_target(clean, noise, rates))
        np.testing.assert_array_equal(original.prediction.get_input_scale(rates),
                                      rebuilt.prediction.get_input_scale(rates))


def test_builtin_autoencoder_record_uses_the_saved_parameters():
    from dew.nn.autoencoders import AutoEncoder, AutoencoderKL, StableDiffusionVAE
    image = jnp.ones((1, 8, 8, 3))
    module = AutoencoderKL(channels=(4,), latent_channels=2, blocks_per_level=1, norm_groups=1,
                           dtype=jnp.float32)
    variables = module.init(jax.random.key(0), image)
    original = StableDiffusionVAE(model=module, params=variables['params'], dtype=jnp.float32)
    rebuilt = AutoEncoder.from_json(original.to_json(), params=original.params)
    np.testing.assert_array_equal(original.encode(original.params, image),
                                  rebuilt.encode(rebuilt.params, image))
    latent = original.encode(original.params, image)
    np.testing.assert_array_equal(original.decode(original.params, latent),
                                  rebuilt.decode(rebuilt.params, latent))


def test_masked_run_returns_its_own_bundle_kind(tmp_path):
    from dew.diffusion.discrete import MDLM
    from dew.interop import Pretrained, PretrainedMaskedDecoder
    from dew.objectives.diffusion.masked import MaskedDiffusionObjective

    masked = model().clone(causal=False, mask_token_id=0, qk_norm=False)
    objective = MaskedDiffusionObjective(masked, MDLM(mask_id=0)(), 8,
                                        head_chunks=1, ema_decay=None, steps=2)
    source = grain.MapDataset.source([{'text': np.arange(1, 9, dtype=np.int32)}] * 8)
    data = Dataset.from_grain(source, batch=8, loading=Loading(workers=0))
    trainer = Trainer(objective, optax.sgd(.01), key=jax.random.key(0),
                      checkpoints=Checkpoints(str(tmp_path / 'run')))
    trainer.fit(data, steps=1, log_every=1, checkpoint_every=1)
    bundle = Pretrained.from_run(tmp_path / 'run')
    assert isinstance(bundle, PretrainedMaskedDecoder)
    assert bundle.model.mask_token_id == 0
