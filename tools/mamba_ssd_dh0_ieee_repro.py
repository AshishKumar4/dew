"""Run the existing SSD regression with only its entered-state-gradient dot changed to IEEE.

This is diagnostic instrumentation: the production file is unchanged. Copy the kernel with a private
globals dictionary so every other SSD dot keeps its original precision and launch configuration.
"""

import sys
import types

import pytest


class IeeeEnteredStateGradient:
    def pytest_configure(self, config):
        # conftest must set the lane's environment before Dew imports JAX. Importing the product at
        # this runner's module scope initialized JAX without JAX_DEFAULT_MATMUL_PRECISION=highest,
        # making the layer's XLA comparison arm use TF32 instead of the ordinary test environment.
        import jax
        from dew.nn.kernels import mamba_ssd

        assert str(jax.config.jax_default_matmul_precision) == "highest"
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
    raise SystemExit(pytest.main(sys.argv[1:], plugins=[IeeeEnteredStateGradient()]))
