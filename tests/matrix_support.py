"""The data matrix's records and their preparation, imported by the matrix's test and by its pool's
workers alike, so a step defined here has one identity in both (`dew.data.plan.StepIdentity`): a
generation the test publishes is the one every worker reads."""

from __future__ import annotations

from pathlib import Path

import numpy as np

MATRIX_RECORDS = 96


def rejects_thirds(row) -> bool:
    """Keep a record unless its id is a multiple of three."""
    return int(row["text"][0]) % 3 != 0


def matrix_prefix(rejecting: bool, *, lengths: bool = False):
    """The data matrix's records, of one to eight tokens each holding its own id, with each one's length
    stated (`_dew_length`, which a published packing plan reads) where `lengths`, and the filter that
    rejects every third where the cell has one."""
    from dew.data import Memory
    from dew.data.pipeline import Pipeline

    records = [{"text": np.full(1 + index % 8, index, np.int32),
                **({"_dew_length": np.int32(1 + index % 8)} if lengths else {})}
               for index in range(MATRIX_RECORDS)]
    pipeline = Pipeline(Memory(records, shard_rows=16))
    if rejecting:
        pipeline = pipeline.filter(rejects_thirds, identity="thirds@1")
    return pipeline


def published_matrix(rejecting: bool, packed: bool, cache_dir: Path):
    """The matrix prefix's rows published once as a sealed generation (`dew.data.compile.publish`),
    with the packing plan of its windows where the cell packs: the preparation that makes a filtered or
    packed read a global order."""
    from dew.data.cache import Cache
    from dew.data.compile import publish

    prefix = matrix_prefix(rejecting, lengths=True)
    return publish(prefix.pack(8, bins=2) if packed else prefix, Cache(str(cache_dir)), read="indexed")
