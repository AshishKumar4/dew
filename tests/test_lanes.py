"""The environment each lane runs the suite under, which conftest sets before
jax opens a backend, checked on dictionaries so no GPU is needed."""

import pytest
from lane_environment import MESH_DEVICES, configure_lane


def test_the_cpu_lane_simulates_the_mesh_devices():
    environ = {}
    configure_lane(environ)
    assert environ["JAX_PLATFORMS"] == "cpu"
    assert environ["XLA_FLAGS"].split() == [
        "--xla_allow_excess_precision=false", f"--xla_force_host_platform_device_count={MESH_DEVICES}"]


@pytest.mark.parametrize("platforms, devices", [("cpu", MESH_DEVICES), ("cuda", 2)])
def test_the_cpu_client_has_a_thread_for_two_launches_of_every_device(platforms, devices):
    """XLA:CPU runs a launch's devices, and the dispatch of the next one, on
    one pool, which on a 4-core runner had a thread per device and no more:
    the lane gives it two per device of its CPU backend, and keeps a run's
    own count."""
    environ = {"JAX_PLATFORMS": platforms, "CUDA_VISIBLE_DEVICES": "0,3"}
    configure_lane(environ)
    assert int(environ["PJRT_NPROC"]) >= 2 * devices
    environ = {"JAX_PLATFORMS": platforms, "CUDA_VISIBLE_DEVICES": "0,3", "PJRT_NPROC": "3"}
    configure_lane(environ)
    assert environ["PJRT_NPROC"] == "3"


@pytest.mark.parametrize("platforms", ["cuda", "cuda,cpu"])
def test_a_cuda_lane_repeats_its_reductions_and_pairs_a_cpu_device_with_each_gpu(platforms):
    """Named alone or beside cpu, cuda gets steps repeatable across processes
    and a CPU backend of one device per visible GPU, after the flags the run
    brought."""
    environ = {"JAX_PLATFORMS": platforms, "CUDA_VISIBLE_DEVICES": "0,3",
               "XLA_FLAGS": "--xla_dump_to=/tmp/dump"}
    configure_lane(environ)
    assert environ["JAX_PLATFORMS"] == "cuda,cpu"
    assert environ["XLA_FLAGS"].split() == [
        "--xla_dump_to=/tmp/dump", "--xla_allow_excess_precision=false", "--xla_gpu_deterministic_ops=true",
        "--xla_gpu_autotune_level=0", "--xla_force_host_platform_device_count=2"]


@pytest.mark.parametrize("platforms", ["tpu", "tpu,cpu"])
def test_a_tpu_lane_keeps_a_cpu_device_beside_each_tpu_device(platforms):
    """A TPU lane keeps the CPU backend, for host callbacks, for float64
    references a TPU cannot compute and for a host layout, which pairs every
    TPU device with a CPU device of its process: one per visible chip here,
    whose generation this machine's PCI bus does not name."""
    environ = {"JAX_PLATFORMS": platforms, "TPU_VISIBLE_CHIPS": "0,1,2,3",
               "XLA_FLAGS": "--xla_dump_to=/tmp/dump"}
    configure_lane(environ)
    assert environ["JAX_PLATFORMS"] == "tpu,cpu"
    assert environ["XLA_FLAGS"].split() == [
        "--xla_dump_to=/tmp/dump", "--xla_allow_excess_precision=false",
        "--xla_force_host_platform_device_count=4"]
