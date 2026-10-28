"""A JEPA case's masking (tests/test_architectures.py's `MASK`): 2x2 target blocks on the architecture
cases' 4x4 grid, apart from the cases, which need no objective."""

from architecture_cases import GRID

from dew.objectives.jepa import MultiBlockMask

# 2x2 target blocks on the 4x4 grid, which leaves 8 context tokens.
MASK = MultiBlockMask.for_grid(GRID, num_targets=2, scale=(0.2, 0.3))
