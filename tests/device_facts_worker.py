"""One rank of a two-process run, each seeing one GPU: the kernel target's facts against the CUDA
driver's own for the GPU the process sees."""

import json
import sys
from pathlib import Path

import jax


def main() -> None:
    rank, coordinator, output = int(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    jax.distributed.initialize(coordinator_address=coordinator, num_processes=2, process_id=rank,
                               initialization_timeout=60)
    from dew.devices import CLOCK_RATE, GLOBAL_MEMORY_BUS_WIDTH, MEMORY_CLOCK_RATE, gpu_attribute
    from dew.nn.kernels.generation import gpu_target

    device = jax.local_devices()[0]
    target = gpu_target()
    clock, memory_clock, bus = (gpu_attribute(0, attribute)
                                for attribute in (CLOCK_RATE, MEMORY_CLOCK_RATE, GLOBAL_MEMORY_BUS_WIDTH))
    output.write_text(json.dumps({
        "processes": jax.process_count(), "id": device.id, "ordinal": device.local_hardware_id,
        "clock_hz": target.clock_hz, "memory_bandwidth": target.memory_bandwidth,
        "multiprocessors": target.multiprocessors, "driver_clock_hz": clock * 1e3,
        "driver_bandwidth": 2 * memory_clock * 1e3 * bus / 8, "driver_multiprocessors": device.core_count}))


if __name__ == "__main__":
    main()
