"""The environment a `dew launch` process pool starts in.

`jax.distributed.initialize` needs the coordinator's address, the process
count and this process's rank. A TPU VM, a Slurm step and an Open MPI launch
leave those in variables jax reads by itself. A plain set of machines leaves
nothing, so `dew.cli.launch` sets the four variables below on every process
and `dew.training.runtime.prepare_process` reads them back. This module is
the contract between the two and imports nothing, so neither side pulls in
the other's dependencies.
"""

COORDINATOR = "JAX_COORDINATOR_ADDRESS"
"""host:port of process 0's coordinator service, which jax reads itself."""
LOCAL_DEVICES = "JAX_LOCAL_DEVICE_IDS"
"""The accelerators a process takes on its host, which jax reads itself."""
PROCESS_COUNT = "DEW_PROCESS_COUNT"
"""How many processes the pool holds; `prepare_process` passes it to jax."""
PROCESS_ID = "DEW_PROCESS_ID"
"""This process's rank in the pool; `prepare_process` passes it to jax."""
