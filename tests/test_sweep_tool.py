"""tools/sweep_grouped_gemm.py's processes (`--processes`): a share whose process dies without its results
stops the record at once, its siblings reaped, never leaving it waiting on the dead one."""

import sys
import time
from pathlib import Path

import numpy as np
import pytest
import sweep_parts

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import sweep_grouped_gemm as sweep


def test_a_share_whose_process_dies_stops_its_record_and_its_siblings():
    """Of three processes, one returns its share's results and two die without them (as one killed out of
    memory does): the record raises `PartDied` within seconds, every process reaped."""
    start = time.monotonic()
    with pytest.raises(sweep.PartDied):
        sweep._timed_apart("shape", "forward", "triton", 0, 3, np.zeros((2, 2), np.float32),
                           part=sweep_parts.returning_or_dying)
    assert time.monotonic() - start < 60
