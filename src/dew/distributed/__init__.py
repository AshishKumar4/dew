"""What the processes of a pool do together on the host: agree a phase or a value before going on, so a
failure on one rank reaches every rank rather than leaving the rest inside a collective (`agreed`,
`agree_process_phase`, `agreed_same`), and bring a global tree to every rank or the first
(`collective_host`), one rank's value to all (`broadcast_from_process_zero`) or every rank's to each
(`from_every_process`). A custom evaluation or training hook that reads data on the host calls these;
Dew's own layers read the same engine (`dew._coordination`). Where a run's arrays live, its mesh
(`MeshSpec`) and its layout over it (`Layout`), is `dew.distributed.placement`'s."""

from dew._coordination import (
    PeerFailure,
    agree_process_phase,
    agreed,
    agreed_same,
    agreement_patience,
    broadcast_from_process_zero,
    collective_host,
    from_every_process,
)
from dew.distributed.placement import Layout, MeshSpec

__all__ = ["Layout", "MeshSpec", "PeerFailure", "agree_process_phase", "agreed", "agreed_same",
           "agreement_patience", "broadcast_from_process_zero", "collective_host", "from_every_process"]
