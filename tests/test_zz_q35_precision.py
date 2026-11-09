"""Scratch operator precision intervention, no product change."""
import jax
import jax.numpy as jnp
import pytest
from dew.nn import linear
from test_zz_q35_cause import test_probe as probe


@pytest.mark.parametrize('stage',['baseline','l2norm','chunk_gated_delta_rule','gated_rms_normalized'])
def test_precision(monkeypatch,stage):
    if stage!='baseline':
        original=getattr(linear,stage)
        count={'l2norm':1,'chunk_gated_delta_rule':5,'gated_rms_normalized':3}[stage]
        def wider(*args,**kwargs):
            if args[0].dtype!=jnp.float32:return original(*args,**kwargs)
            first=list(args[:count]);rest=list(args[count:])
            first=[None if a is None else a.astype(jnp.float64) for a in first]
            if stage=='gated_rms_normalized':rest[1]=jnp.float64
            with jax.default_matmul_precision('highest'):
                result=original(*first,*rest,**kwargs)
            return jax.tree.map(lambda a:a.astype(jnp.float32),result)
        monkeypatch.setattr(linear,stage,wider)
    with jax.enable_x64(True):probe(users=True,seeds=(10,))
