"""Scratch remove chunk algebra only."""
import jax.numpy as jnp
from dew.nn import linear
from dew.nn.kernels.recurrence_chunk import recurrent_delta_rule
from test_zz_q35_cause import test_probe as probe


def test_recurrent(monkeypatch):
    def recurrent(q,k,v,g,b,state=None,*args,**kwargs):
        repeat=v.shape[2]//k.shape[2]
        q,k=(jnp.repeat(x,repeat,axis=2) for x in (q,k))
        return recurrent_delta_rule(q,k,v,g[...,None],b,state,starts=kwargs.get('starts'))
    monkeypatch.setattr(linear,'chunk_gated_delta_rule',recurrent)
    probe()
