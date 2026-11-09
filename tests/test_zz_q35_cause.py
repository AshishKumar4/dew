"""Scratch numerical diagnosis; never a product test."""
import jax
import jax.numpy as jnp
import numpy as np
import test_multimodal_model as T
from reference_error import recording


def test_probe():
    fixture, draw, dew, _ = T.pipeline_arms('qwen35/text')
    stored = recording(fixture)
    lines = []
    narrow = jax.jit(lambda inputs: dew(jnp.float32, inputs)['updated_logits'])
    wide = jax.jit(lambda inputs: dew(jnp.float64, inputs)['updated_logits'])
    for seed in (10, 700):
        inputs = draw(seed)
        a = np.asarray(narrow(inputs), np.float64)
        with jax.enable_x64(True):
            b = np.asarray(wide(inputs))
        e = (a-b)**2
        pos = np.sum(e, -1)
        k = np.unravel_index(pos.argmax(), pos.shape)
        keys = [k for k in stored if 'updated_logits' in k]
        mse = stored['multimodal-qwen35-text.mse.updated_logits'][seed]
        lines.append('seed %d mse %.12g ref %.12g ratio %.8g peak %s fraction %.8g maxerr %.8g keys %s' % (seed,e.mean(),mse,np.sqrt(e.mean()/mse), k,pos[k]/pos.sum(),np.abs(a-b).max(),keys[:3]))
    raise AssertionError('PROBE\n'+'\n'.join(lines))
