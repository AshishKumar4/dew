"""The environment each lane runs the suite under, which conftest sets before
jax opens a backend, checked on dictionaries so no GPU is needed."""

import pytest
from conftest import MESH_DEVICES, configure_lane


def test_the_cpu_lane_simulates_the_mesh_devices():
    environ = {}
    configure_lane(environ)
    assert environ["JAX_PLATFORMS"] == "cpu"
    assert environ["XLA_FLAGS"].split() == [f"--xla_force_host_platform_device_count={MESH_DEVICES}"]


@pytest.mark.parametrize("platforms", ["cuda", "cuda,cpu"])
def test_a_cuda_lane_repeats_its_reductions_and_pairs_a_cpu_device_with_each_gpu(platforms):
    """Named alone or beside cpu, cuda gets repeatable reductions and a CPU
    backend of one device per visible GPU, after the flags the run brought."""
    environ = {"JAX_PLATFORMS": platforms, "CUDA_VISIBLE_DEVICES": "0,3",
               "XLA_FLAGS": "--xla_dump_to=/tmp/dump"}
    configure_lane(environ)
    assert environ["JAX_PLATFORMS"] == "cuda,cpu"
    assert environ["XLA_FLAGS"].split() == [
        "--xla_dump_to=/tmp/dump", "--xla_gpu_deterministic_ops=true",
        "--xla_force_host_platform_device_count=2"]
