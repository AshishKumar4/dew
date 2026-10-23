"""Unmodified recurrence tests, with the two-scan chunk-parallel diagnostic adjoint."""
import pytest
import test_recurrence_chunk as token
import test_recurrence_stability as stability
from probe_aligned_chunk_adjoint_nosysquid import chunk_adjoint_wy_source

from dew.nn.kernels import recurrence_chunk

for _module in (token, stability):
    globals().update({name: value for name, value in vars(_module).items() if name.startswith("test_")})


@pytest.fixture(autouse=True)
def diagnostic_adjoint(monkeypatch):
    monkeypatch.setattr(recurrence_chunk, "chunk_delta_rule", chunk_adjoint_wy_source)
