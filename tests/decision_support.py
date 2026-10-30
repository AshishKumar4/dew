"""Decision rows as a run reads them, for the tests of decision objectives and their packing."""

from __future__ import annotations


def decision_rows(examples, batch: int | None, validation=None, *, objective=None, tokens: int | None = None):
    """`examples` as a run's decision batches of `batch` rows (`dew.decision.decisions`), or of `tokens`
    tokens, read in memory, with `validation`'s examples as its held-out split, compiled with
    `objective`'s stages (`dew.data.compile.compile`)."""
    from dew.data import Memory
    from dew.data.compile import compile
    from dew.decision import decisions

    def chain(rows):
        parsed = Memory([{"example": example} for example in rows]).pipe(decisions)
        return parsed.batch(tokens=tokens) if tokens is not None else parsed.batch(batch)

    held = None if validation is None else chain(validation)
    return compile(chain(examples), validation=held, read="memory", objective=objective)
