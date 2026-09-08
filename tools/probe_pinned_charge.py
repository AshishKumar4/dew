#!/usr/bin/env python3
"""Is pinned host memory charged where a budget can hold it, and what does a
banked load peak at.

Two parts, the second gated on the first. The sentinel allocates a device
donor, baselines the control group after the backend and that donor exist, and
copies the donor device to host into pinned memory: only that copy may move
`memory.current`, and no host-side buffer takes part, so a NumPy allocation
cannot answer for it. If the move is absent, partial or unexplained the probe
stops and says so, because a budget that does not see the allocation is not a
budget. Only if it is charged does the matrix run: one small store built with
the copies of each bank waited for, and again with them all queued, at three
bank sizes.

Run it inside a control group that caps the whole process tree, since the
point is what the tree costs:

    systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 -- \\
        env JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src \\
        python tools/probe_pinned_charge.py --store-mib 512
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tyro
from jax.sharding import NamedSharding, PartitionSpec as P

MIB = 1024 * 1024
SENTINEL_MIB = 256
# The copy has to move the group's charge by its own size to count as charged.
# A tenth is slack for whatever else moves between two reads.
TOLERANCE = 0.1


CAP_BYTES = 4 * 1024 ** 3
HEADROOM_BYTES = 12 * 1024 ** 3


def preflight(store_mib: int | None = None, bank_counts=None) -> dict:
    """Refuse to start unless a live budget already bounds this process tree.

    Read before the backend is opened, because a check that runs after the
    first allocation is not a check. Reported limits are not taken as a
    guard: the group has to be an active cgroup v2 memory controller with a
    finite `memory.max` no larger than the agreed cap and swap turned off, and
    the machine has to have the agreed headroom left. Anything missing,
    unlimited or unreadable exits nonzero with the reason, since running
    without enforcement is what the cap exists to prevent.
    """
    line = Path("/proc/self/cgroup").read_text().strip().splitlines()[-1]
    where = Path("/sys/fs/cgroup") / line.split(":")[-1].lstrip("/")
    if not (where / "memory.current").is_file():
        raise SystemExit(
            f"refusing to run: {where} is not an active cgroup v2 memory controller, "
            f"so nothing bounds this process tree")
    for name in ("memory.max", "memory.swap.max"):
        if not (where / name).is_file():
            raise SystemExit(f"refusing to run: {where}/{name} is not readable")
    limit = (where / "memory.max").read_text().strip()
    if limit == "max":
        raise SystemExit(
            f"refusing to run: {where}/memory.max is unlimited; start inside a scope "
            f"with MemoryMax set, at most {CAP_BYTES} bytes")
    if int(limit) > CAP_BYTES:
        raise SystemExit(
            f"refusing to run: {where}/memory.max is {limit}, over the agreed "
            f"{CAP_BYTES} byte cap")
    swap = (where / "memory.swap.max").read_text().strip()
    if swap != "0":
        raise SystemExit(
            f"refusing to run: {where}/memory.swap.max is {swap}, not 0, so the cap "
            f"can be paid for in swap")
    available = 0
    for entry in Path("/proc/meminfo").read_text().splitlines():
        if entry.startswith("MemAvailable:"):
            available = int(entry.split()[1]) * 1024
    if available < HEADROOM_BYTES:
        raise SystemExit(
            f"refusing to run: MemAvailable is {available} bytes, under the "
            f"{HEADROOM_BYTES} byte headroom this is allowed to leave")
    if store_mib is not None and not 0 < store_mib <= 512:
        raise SystemExit(
            f"refusing to run: store_mib is {store_mib}, outside the permitted "
            f"1 to 512 MiB")
    if bank_counts is not None and any(count < 1 for count in bank_counts):
        raise SystemExit(f"refusing to run: bank_counts {list(bank_counts)} is not positive")
    return {"cgroup": str(where), "memory.max": limit, "memory.swap.max": swap,
            "MemAvailable": available}


@dataclass(frozen=True)
class ProbeConfig:
    """How large a store to build, and in how many banks."""

    store_mib: int = 512
    bank_counts: list[int] | None = None
    out: str | None = None


def cgroup() -> Path | None:
    """This process's control group directory, or None outside cgroup v2."""
    line = Path("/proc/self/cgroup").read_text().strip().splitlines()[-1]
    relative = line.split(":")[-1].lstrip("/")
    path = Path("/sys/fs/cgroup") / relative
    return path if (path / "memory.current").is_file() else None


def limits(where: Path) -> dict:
    return {name: (where / name).read_text().strip()
            for name in ("memory.max", "memory.swap.max")
            if (where / name).is_file()}


def charged(where: Path | None) -> dict:
    """What the group and the kernel say this tree holds right now."""
    values: dict[str, int] = {}
    if where is not None:
        for name in ("memory.current", "memory.peak", "memory.swap.current"):
            if (where / name).is_file():
                values[name] = int((where / name).read_text().strip())
        if (where / "memory.events").is_file():
            for line in (where / "memory.events").read_text().splitlines():
                key, count = line.split()
                values[f"events.{key}"] = int(count)
    status = Path("/proc/self/status").read_text()
    for line in status.splitlines():
        name, _, rest = line.partition(":")
        if name in ("VmRSS", "VmHWM", "VmSwap"):
            values[name] = int(rest.split()[0]) * 1024
    rollup = Path("/proc/self/smaps_rollup")
    if rollup.is_file():
        for line in rollup.read_text().splitlines():
            name, _, rest = line.partition(":")
            if name in ("Rss", "Pss"):
                values[f"smaps.{name}"] = int(rest.split()[0]) * 1024
    return values


def spaces(mesh):
    device = NamedSharding(mesh, P())
    return device, device.with_memory_kind("pinned_host")


def sentinel(where: Path | None, mesh) -> dict:
    """The device-donor charge check, and whether the matrix may run."""
    device, host = spaces(mesh)
    elements = SENTINEL_MIB * MIB // 4
    started = charged(where)
    donor = jax.block_until_ready(
        jax.device_put(jnp.zeros((elements,), jnp.float32), device))
    after_donor = charged(where)
    copy = jax.block_until_ready(jax.device_put(donor, host))
    after_copy = charged(where)
    assert copy.sharding.memory_kind == "pinned_host"
    wanted = SENTINEL_MIB * MIB
    moved = after_copy.get("memory.current", 0) - after_donor.get("memory.current", 0)
    reserved = after_donor.get("memory.current", 0) - started.get("memory.current", 0)
    verdict = {
        "cgroup": None if where is None else str(where),
        "limits": {} if where is None else limits(where),
        "after_init": started,
        "after_device_donor": after_donor,
        "after_host_copy": after_copy,
        "sentinel_bytes": wanted,
        "allocator_reservation_bytes": reserved,
        "charged_bytes": moved,
        "charged": bool(where is not None and abs(moved - wanted) <= TOLERANCE * wanted),
    }
    del donor, copy
    return verdict


def load(where: Path | None, mesh, store_mib: int, banks: int, serialized: bool) -> dict:
    """One store of `banks` banks, waiting for each or queueing them all."""
    device, host = spaces(mesh)
    per_bank = store_mib * MIB // banks
    elements = per_bank // 2
    started = charged(where)
    placed = []
    source_bytes = 0
    for index in range(banks):
        rows = np.full((elements,), np.float32(index + 1), dtype=np.dtype(jnp.bfloat16))
        source_bytes += rows.nbytes
        landed = jax.device_put(rows, host)
        if serialized:
            jax.block_until_ready(landed)
        placed.append(landed)
        del rows
    jax.block_until_ready(placed)
    settled = charged(where)
    destination_bytes = sum(
        shard.data.nbytes for leaf in placed for shard in leaf.addressable_shards)
    del placed
    return {
        "banks": banks, "serialized": serialized,
        "bank_bytes": per_bank, "source_bytes_read": source_bytes,
        "destination_bytes_local": destination_bytes,
        "before": started, "after": settled,
        "current_growth": settled.get("memory.current", 0) - started.get("memory.current", 0),
        "peak_over_destination": (settled.get("memory.peak", 0) / destination_bytes
                                  if destination_bytes else None),
    }


def main(config: ProbeConfig) -> None:
    enforced = preflight(config.store_mib, config.bank_counts)
    where = cgroup()
    mesh = jax.make_mesh((1,), ("x",), devices=jax.devices()[:1])
    report = {"enforced": enforced,
              "devices": [str(device) for device in jax.devices()],
              "sentinel": sentinel(where, mesh)}
    if not report["sentinel"]["charged"]:
        report["matrix"] = None
        report["refused"] = (
            "the control group did not see the pinned copy, so a cap over this tree "
            "would not bound it; the matrix was not run" if where is not None else
            "no cgroup v2 memory controller on this process, so nothing enforces a "
            "tree budget; the matrix was not run")
    else:
        counts = config.bank_counts or [1, 4, 16]
        report["matrix"] = [
            load(where, mesh, config.store_mib, banks, serialized)
            for banks in counts for serialized in (True, False)]
    print(json.dumps(report, indent=1, default=float))
    if config.out is not None:
        Path(config.out).write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main(tyro.cli(ProbeConfig))
