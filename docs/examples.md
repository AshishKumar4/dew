# Example scripts

The `examples/` folder holds complete programs, each a single file you can copy and change. The descriptions below are the scripts' own docstrings, so they match the code in the repository. Every script has a command-line interface; run it with `--help` to see the options.

The first group trains one kind of model on one dataset and is the quickest way to see a whole workflow. The second group runs a full job, from data to scored or exported weights. Each of those takes a `--smoke` flag that swaps the real settings for the repository's tiny fixtures, a few steps and one CPU device, which is how the test suite runs them. [End-to-end runs](guides/end-to-end.md) gives both command lines for each.
