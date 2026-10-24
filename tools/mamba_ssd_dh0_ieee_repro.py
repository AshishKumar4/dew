"""Run the existing SSD regression with only its entered-state-gradient dot changed to IEEE.

This is diagnostic instrumentation: the production file is unchanged. Copy the kernel with a private
globals dictionary so every other SSD dot keeps its original precision and launch configuration.
"""

import sys
import types

import pytest

from dew.nn.kernels import mamba_ssd


original_dot = mamba_ssd._dot


def ieee_dot(a, b, *, exact, **kwargs):
    return original_dot(a, b, exact=exact, ieee=True)


original = mamba_ssd._dstates_kernel
instrumented = types.FunctionType(
    original.__code__, {**original.__globals__, "_dot": ieee_dot},
    original.__name__, original.__defaults__, original.__closure__)
instrumented.__kwdefaults__ = original.__kwdefaults__
mamba_ssd._dstates_kernel = instrumented

if __name__ == "__main__":
    raise SystemExit(pytest.main(sys.argv[1:]))
