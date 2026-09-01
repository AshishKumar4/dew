"""Device mesh, parameter sharding and host-to-device prefetch."""

import queue
import threading
from typing import Iterator, Optional

import jax
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

DATA_AXIS = 'data'
FSDP_AXIS = 'fsdp'

# Batches are split across every device, whichever axis it sits on; only
# parameters distinguish the two axes.
BATCH_SPEC = P((DATA_AXIS, FSDP_AXIS))

# Below this many elements a parameter costs more in collectives than it saves
# in memory, so it stays replicated.
DEFAULT_MIN_SHARD_SIZE = 2 ** 16


def build_mesh(fsdp_size: int = 1, devices: Optional[list] = None) -> Mesh:
    """Two-axis device mesh: parameters shard over 'fsdp', replicate over 'data'.

    fsdp_size=1 degenerates to plain data parallelism, so the same code path
    serves both without a flag. Axes are Auto so GSPMD infers the collectives
    rather than us writing them by hand.
    """
    devices = list(devices) if devices is not None else jax.devices()
    if fsdp_size < 1 or len(devices) % fsdp_size:
        raise ValueError(
            f"fsdp_size {fsdp_size} must be a positive divisor of device count {len(devices)}")
    return jax.make_mesh(
        (len(devices) // fsdp_size, fsdp_size),
        (DATA_AXIS, FSDP_AXIS),
        devices=devices,
        axis_types=(AxisType.Auto, AxisType.Auto),
    )


def parameter_spec(shape: tuple, fsdp_size: int, min_shard_size: int) -> P:
    """Shard the largest evenly-divisible axis over 'fsdp', else replicate.

    Applied to every leaf of the train state, not just params: optimizer moments
    and EMA copies have the same shapes as the params they track, so they pick
    up the same spec without anyone having to describe the optimizer's layout.
    """
    if fsdp_size == 1 or int(np.prod(shape, dtype=np.int64)) < min_shard_size:
        return P()
    for axis in sorted(range(len(shape)), key=lambda i: -shape[i]):
        if shape[axis] % fsdp_size == 0:
            return P(*([None] * axis), FSDP_AXIS)
    return P()


def state_sharding_tree(
    mesh: Mesh, abstract_state, min_shard_size: int = DEFAULT_MIN_SHARD_SIZE
):
    """Map a train state of ShapeDtypeStructs to its NamedSharding tree."""
    fsdp_size = mesh.shape[FSDP_AXIS]
    return jax.tree.map(
        lambda x: NamedSharding(mesh, parameter_spec(x.shape, fsdp_size, min_shard_size)),
        abstract_state,
    )


def batch_sharding(mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, BATCH_SPEC)


def shard_batch(sharding: NamedSharding, batch):
    """Assemble this process's slice of each array into a globally sharded one."""
    return jax.tree.map(
        lambda x: jax.make_array_from_process_local_data(sharding, np.asarray(x)), batch)


class DevicePrefetchIterator:
    """Runs the host-to-device batch transfer a few batches ahead of the loop.

    Without this the transfer sits on the critical path between steps, because
    the loop only starts moving batch N+1 after step N has been dispatched.
    """

    def __init__(self, iterator: Iterator, sharding: NamedSharding, depth: int = 2,
                 source_state=None):
        self._iterator = iter(iterator)
        self._sharding = sharding
        self._queue = queue.Queue(maxsize=depth)
        self._terminal: Optional[BaseException] = None
        self._checkpointable = hasattr(self._iterator, 'get_state')
        if source_state is not None:
            if not self._checkpointable:
                raise TypeError(
                    f"{type(self._iterator).__name__} cannot resume from a saved position")
            self._iterator.set_state(source_state)
        # Position of the source iterator as of the batch most recently handed
        # out, so a checkpoint resumes at the next unseen batch rather than at
        # whatever the prefetch thread has already raced ahead to.
        self.source_state = source_state
        self._thread = threading.Thread(target=self._prefetch, daemon=True)
        self._thread.start()

    def _prefetch(self):
        try:
            while True:
                batch = next(self._iterator)
                state = self._iterator.get_state() if self._checkpointable else None
                self._queue.put((shard_batch(self._sharding, batch), state))
        except StopIteration:
            self._queue.put(StopIteration())
        except BaseException as error:  # surfaced on the consumer's thread
            self._queue.put(error)

    def __iter__(self):
        return self

    def __next__(self):
        if self._terminal is not None:
            raise self._terminal
        item = self._queue.get()
        if isinstance(item, BaseException):
            self._terminal = item
            raise item
        batch, self.source_state = item
        return batch
