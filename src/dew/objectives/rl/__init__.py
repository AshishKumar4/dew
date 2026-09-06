"""Online RL objectives: rollouts, preference losses and group updates.

These compose the array math in `dew.rl`; the import gate in
tests/test_rl_imports.py keeps that arrow one way. `dew.rl` may read `dew`,
and nothing under `dew` outside these two packages may read `dew.rl`.
"""

from dew.data.preferences import IDS_KEY, MASK_KEY

from .preference import DPOObjective
from .rollout import Reward, SampledRollout

__all__ = ["DPOObjective", "IDS_KEY", "MASK_KEY", "Reward", "SampledRollout"]
