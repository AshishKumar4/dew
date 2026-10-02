"""Python training saves the model/task declaration with its own checkpoint."""
import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np
import optax

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
    trainer.fit(data, steps=2, log_every=2, checkpoint_every=1)
    checkpoints.wait()
    assert not (tmp_path / 'run' / 'run.json').exists()
    record = Checkpoints(str(tmp_path / 'run')).artifact(2)
    assert record['objective'] == 'lm'
    assert record['seq_len'] == 8
    rebuilt = ModelConfig.from_dict(record['model']).build()
    assert rebuilt == objective.model.clone(dtype=jnp.float32)
    assert record['tokenizer'] is None
