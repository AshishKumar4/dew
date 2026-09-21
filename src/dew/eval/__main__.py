"""`python -m dew.eval` is lm_eval's command line with `dew` registered.

lm_eval 0.4 discovers models only through its own registry, which a third
party fills by being imported, and its console script imports nothing of
ours. Importing `dew.eval.harness` here is that import, and the rest is
lm_eval's own entry point, so every flag and every task is the harness's.
"""

from __future__ import annotations

import sys


def main() -> int:
    from lm_eval.__main__ import cli_evaluate

    import dew.eval.harness  # noqa: F401 registers the `dew` model with lm_eval

    sys.argv = ["lm_eval", *sys.argv[1:]]
    cli_evaluate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
