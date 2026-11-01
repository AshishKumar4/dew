"""The room NCCL's communicators take outside XLA's pool (ledger 2901): measured once per configuration by a
child process before the backend opens and kept in Dew's cache, the pool's share sized to leave it, a share
the user set too high for it refused naming the bytes, and a compiled step whose collectives need more room
than its GPUs have outside the pool refused before it runs, never left to stall in a collective. The
measurement itself needs two GPUs, which a two-GPU run of the check makes; here it is the child's report, as the
parent reads it."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from dew.training import communicators
from dew.training.communicators import CommunicatorRoom, Needs

SOURCE = Path(__file__).resolve().parents[1] / "src"
GIB = 2**30

NEEDS = Needs(communicators={0: {0: GIB, 1: GIB}, 1: {0: 2 * GIB, 1: 2 * GIB}},
              runtime={0: GIB // 2, 1: GIB // 2}, seconds=4.0)
"""Two GPUs' measurement: a collective's communicator takes 1 GiB on each, a permute's 2, the runtime half."""

MEMORY = {0: (79 * GIB, 80 * GIB), 1: (76 * GIB, 80 * GIB)}
"""Each GPU's free and total bytes before the backend opens: 79 and 76 of 80 GiB free."""


def test_dews_share_leaves_the_communicators_room_and_the_users_is_held_to_it():
    """Dew's share of each GPU leaves the room its step's communicators and the runtime take outside the
    pool, the tightest GPU deciding, never above Dew's 95%; a share the user set within it is kept, and one
    beyond it is refused naming each side's bytes."""
    share = communicators.pool_share(NEEDS, MEMORY, requested=None, kinds=frozenset({0}))
    # The tighter GPU decides: 76 GiB free less 1.5 GiB of room, to a thousandth below.
    assert share * 80 * GIB <= 76 * GIB - GIB - GIB // 2 < (share + 0.001) * 80 * GIB
    roomy = {0: (79 * GIB, 80 * GIB)}
    assert communicators.pool_share(NEEDS, roomy, requested=None, kinds=frozenset({0})) == 0.95
    assert communicators.pool_share(NEEDS, MEMORY, requested=0.9, kinds=frozenset({0})) == 0.9
    with pytest.raises(CommunicatorRoom) as refused:
        communicators.pool_share(NEEDS, MEMORY, requested=0.97, kinds=frozenset({0, 1}))
    assert refused.value.needed == {0: 3 * GIB + GIB // 2, 1: 3 * GIB + GIB // 2}
    assert refused.value.room == {0: 79 * GIB - int(0.97 * 80 * GIB), 1: 76 * GIB - int(0.97 * 80 * GIB)}


def test_a_measurement_is_paid_once_per_configuration(tmp_path, monkeypatch):
    """A configuration's measurement is kept in Dew's cache and read back without a child process; another
    NCCL setting is another configuration, measured afresh."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    taken = []

    def measure(configuration):
        taken.append(configuration)
        return NEEDS

    configuration = {"gpus": ["NVIDIA H100 80GB HBM3"] * 2, "driver": 12090, "nccl": 23203, "nccl_env": {}}
    assert communicators.measured(configuration, measure=measure) == NEEDS
    assert communicators.measured(configuration, measure=measure) == NEEDS
    other = {**configuration, "nccl_env": {"NCCL_NVLS_ENABLE": "0"}}
    assert communicators.measured(other, measure=measure) == NEEDS
    assert taken == [configuration, other]


def test_a_compiled_steps_collectives_are_held_to_the_room_outside_the_pool():
    """A compiled step whose collectives (a collective's and a permute's, the communicators XLA keys apart)
    need more than its GPUs have free outside the pool is refused before it runs, naming the bytes; one
    within it passes."""
    kinds = communicators.kinds_of("%x = f32[8] all-gather(...)\n%y = f32[8] collective-permute(...)")
    assert kinds == frozenset({0, 1})
    assert communicators.kinds_of("%x = f32[8] add(...)") == frozenset()
    communicators.checked_room({0: 4 * GIB, 1: 4 * GIB}, NEEDS, kinds)
    with pytest.raises(CommunicatorRoom) as refused:
        communicators.checked_room({0: 4 * GIB, 1: 2 * GIB}, NEEDS, kinds)
    assert refused.value.needed[1] == 3 * GIB and refused.value.room[1] == 2 * GIB


def test_a_host_of_one_gpu_or_none_measures_nothing():
    """A host with fewer than two GPUs makes no communicator, so it measures nothing and its share stays
    Dew's own."""
    assert communicators.needs_of(gpus=1) is None
    assert communicators.needs_of(gpus=0) is None


@pytest.mark.parametrize(("set_by_user", "expected"), [(None, "0"), ("1", "1")])
def test_import_dew_connects_nccls_peers_at_creation_unless_told_otherwise(set_by_user, expected):
    """`import dew` has NCCL connect every peer as it creates a communicator (NCCL_RUNTIME_CONNECT=0), where
    XLA returns an allocation's failure to every rank; a value the user set is theirs."""
    env = {key: value for key, value in os.environ.items() if key != "NCCL_RUNTIME_CONNECT"}
    if set_by_user is not None:
        env["NCCL_RUNTIME_CONNECT"] = set_by_user
    script = "import json, os\nimport dew\nprint(json.dumps(os.environ.get('NCCL_RUNTIME_CONNECT')))\n"
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True,
                          env={**env, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(SOURCE)})
    assert json.loads(done.stdout.strip().splitlines()[-1]) == expected
