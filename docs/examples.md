# Example scripts

The `examples/` folder holds complete programs, one file each. The descriptions below are generated from the scripts' docstrings. Every script but one has a command-line interface, and `--help` lists its options. `train_supervised.py` is a Python experiment, which `dew train` runs.

The first group trains one kind of model on one dataset. The second group runs a full job, from data to scored or exported weights. Each script in the second group takes `--smoke`, which replaces the real settings with the repository's tiny fixtures, a few steps and one CPU device; the test suite runs them that way. [End-to-end runs](guides/end-to-end.md) gives both command lines for each.
