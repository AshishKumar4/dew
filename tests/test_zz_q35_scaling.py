"""Scratch query scaling placement control."""
from dew.nn.kernels import recurrence_chunk as R
from test_zz_q35_cause import test_probe as probe


def test_scaling(monkeypatch):
    original=R.xla_mainloop
    def scaled(config,q,k,v,g,b,state=None,starts=None,*,scale,**kwargs):
        return original(config,q*scale,k,v,g,b,state,starts,scale=1.0,**kwargs)
    monkeypatch.setattr(R,'xla_mainloop',scaled)
    probe()
