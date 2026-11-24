"""Falcon's fused MQA/MHA and one-norm parallel residual against transformers 5.16.1."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from model_support import attending

from dew.interop import Pretrained

ROOT = Path(__file__).parent / 'fixtures' / 'hf'



@attending("reference")
def test_falcon_source_export_writes_updated_fused_projections(tmp_path):
    from dataclasses import replace

    loaded = Pretrained.load(ROOT / 'falcon-tiny', dtype='float32')
    variables = jax.tree.map(jnp.asarray, loaded.variables)
    variables['params']['layers_0']['self_attn']['q_proj']['kernel'] *= 2
    replace(loaded, variables=variables).save(tmp_path)
    restored = Pretrained.load(tmp_path, dtype='float32')
    ids = np.load(ROOT / 'falcon-tiny' / 'input_ids.npy')
    expected = loaded.model.apply(variables, ids)
    np.testing.assert_array_equal(restored.model.apply(restored.variables, ids), expected)
