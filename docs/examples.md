# Example scripts

The `examples/` folder holds complete programs, one file each. The descriptions below are generated from the scripts' docstrings. Every script has a command-line interface; `--help` lists its options.

The first group trains one kind of model on one dataset. The second group runs a full job, from data to scored or exported weights. Each script in the second group takes `--smoke`, which replaces the real settings with the repository's tiny fixtures, a few steps and one CPU device; the test suite runs them that way. [End-to-end runs](guides/end-to-end.md) gives both command lines for each.
