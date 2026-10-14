# Data-pipeline migration receipts

These scripts produced the evidence for the move to `Pipeline` recipes: what the old loaders read. Each one is pinned to the revisions it names, so it reproduces that evidence and is not a maintained tool or a CI gate. Run them from the repository root, with Dew installed from the checkout. Their `r2://` pins name Dew's own data bucket, which public Dew does not reach. Reading them needs the dew-ops `r2` filesystem adapter, and the publishing steps need write access through it.

| File | Baseline | Purpose | Reproduction |
| --- | --- | --- | --- |
| `data_oracle.py`, `data_oracle_legacy.py` | Dew `66beb5305` | Records what each old loader read from synthetic records, at 0 and 2 workers and 1 and 2 shares. `data_oracle.py` downloads the baseline and runs `data_oracle_legacy.py` inside it. The gzipped output is `tests/fixtures/golden/loaders-66beb5305.json.gz`, which `tests/test_pipeline_golden.py` reads. | `python tools/measurements/data-pipeline/data_oracle.py --output loaders-66beb5305.json` |
