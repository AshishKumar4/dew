"""Decision rows as a run reads them, for the tests of decision objectives and their packing."""

from __future__ import annotations


def decision_rows(examples, batch: int | None, validation=None, *, objective=None, tokens: int | None = None,
                  plan_version: int | None = None):
    """`examples` as a run's decision batches of `batch` rows (`dew.decision.decisions`), or of `tokens`
    tokens, read in memory, with `validation`'s examples as its held-out split, compiled with
    `objective`'s stages (`dew.data.compile.compile`), read by data plan `plan_version` where given, as a
    run saved under it reads them, else this Dew's."""
    from dew.data import Memory
    from dew.data.compile import compile
    from dew.data.plan import PLAN_VERSIONS
    from dew.decision import decisions

    def chain(rows):
        parsed = Memory([{"example": example} for example in rows]).pipe(decisions)
        return parsed.batch(tokens=tokens) if tokens is not None else parsed.batch(batch)

    held = None if validation is None else chain(validation)
    return compile(chain(examples), validation=held, read="memory", objective=objective,
                   plan_version=PLAN_VERSIONS[-1] if plan_version is None else plan_version)
