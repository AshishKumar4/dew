"""Cases the gate's own tests share (tests/test_reference_error.py, tests/test_check1_judge.py): a product's
draws, a leaf of rows that cancel, a failure's findings as data, and a site run as armada's map tasks and
their judge."""

import jax.numpy as jnp
import numpy as np

from dew.parity import GateFailure


def cancelling(split: bool, *, leak: bool = False, residue: float = 0.0):
    """A leaf of rows that cancel, rows of no terms and plain rows: rows 0-7 a dot less itself (the
    reference's the same dot twice, exactly zero; split, Dew's second dot summed in two halves, an ulp
    or so off zero; leaked, one term dropped from it), rows 8-11 the dot of zero rows (`residue` added),
    and the rest the plain dot, every arm alike."""
    def halves(x, w):
        return x[:, :32] @ w[:32] + (x[:, 33:] @ w[33:] if leak else x[:, 32:] @ w[32:])

    def arm(x, w):
        whole = x @ w
        other = halves(x, w) if split else x @ w
        return {"rows": jnp.concatenate([whole[:8] - other[:8], whole[8:12] + residue, whole[12:]])}
    return arm


def cancelling_draw(seed):
    """A [64, 64] by [64, 4] dot's operands, rows 8-11 of the first zero."""
    rng = np.random.default_rng([2440, seed])
    x = rng.standard_normal((64, 64))
    x[8:12] = 0
    return jnp.asarray(x, jnp.float32), jnp.asarray(rng.standard_normal((64, 4)), jnp.float32)


def found(failure: GateFailure) -> list[tuple[str, str, float]]:
    """A gate failure's findings as data: each one's leaf, rule and measure (`Finding`), not its wording."""
    return [(finding.leaf, finding.rule, finding.measure) for finding in failure.findings]


def sharded_site(monkeypatch, root, shards, site):
    """`site()` run as armada runs a sharded site: a map task for each range of `shards` (DRAWS), its
    records under `root` (RECORDS), then their judge (POOL over `root`); what each map task's run
    returned and what the judge's did."""
    from check1_wire import DRAWS, POOL, RECORDS

    drawn = []
    for index, (start, stop) in enumerate(shards):
        monkeypatch.setenv(DRAWS, f"{start}:{stop}")
        monkeypatch.setenv(RECORDS, str(root / str(index)))
        drawn.append(site())
    monkeypatch.delenv(DRAWS)
    monkeypatch.delenv(RECORDS)
    monkeypatch.setenv(POOL, str(root))
    try:
        return drawn, site()
    finally:
        monkeypatch.delenv(POOL)


def product_draws(key: int, inner: int):
    """Draws of a [16, `inner`] by [`inner`, 8] fp32 product's operands, a stream of their own by `key`."""
    def draw(seed):
        rng = np.random.default_rng([key, seed])
        return (jnp.asarray(rng.standard_normal((16, inner)), jnp.float32),
                jnp.asarray(rng.standard_normal((inner, 8)), jnp.float32))
    return draw


a_product_draw = product_draws(2641, 64)
"""A [16, 64] by [64, 8] fp32 product's operands."""
