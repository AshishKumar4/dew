"""Diagnostic only: replace local readout dL's CPU matrix reduction by column GEMVs."""
import jax

import test_algebra_scan as original_test
from dew.algebra import scan
from dew.nn.precision import float32_matmul


def test_deep_decay_dc_columns(monkeypatch):
    original = scan._issued

    @jax.custom_vjp
    def local(left, right):
        return original("readout_local", left, right)

    def forward(left, right):
        return original("readout_local", left, right), (left, right)

    def backward(saved, d):
        left, right = saved
        dl = jax.lax.map(lambda column: float32_matmul(d, column), right).T
        dr = float32_matmul(left[::-1].T, d[::-1])
        return dl, dr

    local.defvjp(forward, backward)

    def issued(name, left, right):
        return local(left, right) if name == "readout_local" else original(name, left, right)

    monkeypatch.setattr(scan, "_issued", issued)
    original_test.test_the_ssd_chunks_adjoint_is_the_token_scans_gradient(original_test.deep_draw)
