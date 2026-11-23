"""One rank of a two-process CPU run: the process whose device a launch traced outside any mesh and any
run's target is certified on (`report._certifying_device`)."""

import json
import sys
from pathlib import Path

import jax


def main() -> None:
    rank, coordinator, output = int(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    jax.distributed.initialize(coordinator_address=coordinator, num_processes=2, process_id=rank,
                               initialization_timeout=60)
    from dew.nn.kernels.templates import report

    device = report._certifying_device()
    output.write_text(json.dumps({"process": jax.process_index(), "devices": jax.device_count(),
                                  "certifying": None if device is None else device.process_index}))


if __name__ == "__main__":
    main()
