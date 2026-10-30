"""A training restart shares native artifacts without reviving a refused spilling launch."""

import pytest
from generation_support import lowers
from test_kernel_spills import (
    a_cached_spilling_run_trains_its_stepped_program,
    fresh_compilation_nonce as fresh_compilation_nonce,
)

pytestmark = pytest.mark.gpu(reason="three H100 training processes share a native compilation cache")


@lowers("sm90")
def test_a_training_restart_runs_the_stepped_native_artifact(tmp_path, nonce):
    a_cached_spilling_run_trains_its_stepped_program(tmp_path, nonce)
