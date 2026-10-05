# Example scripts

Each file in `examples/` is a complete program with a command-line interface. Use `--help` to see its options. The descriptions below come from the scripts' docstrings.

The first group trains one kind of model on one dataset. The second runs a full job from data to scored or exported weights. Scripts in the second group accept `--smoke` to use the repository's tiny fixtures, a few steps and one CPU device. The test suite uses these settings. [End-to-end runs](guides/end-to-end.md) gives both full and smoke commands for each script.
