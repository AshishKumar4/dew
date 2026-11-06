"""Check 2's native step witness for the Mosaic attention forward (tests/attention_witness.py, row 2084 (f)),
on Mosaic GPU's interpreter: the certify launch's states and output inside their step enclosures (nulls),
and each fault the witness exists for refused at its elements: a dropped rescale, a maximum from the wrong
tile, sums rounded to bf16 where the walk holds fp32, and a mask off by one key."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from attention_oracle import kept
from attention_witness import witnessed

SHAPE = (1, 256, 2, 64)


def operands(seed: int):
    """bf16 queries, keys and values `[B, S, H, D]`, drawn; the keys scaled so the rows' maxima move across
    tiles."""
    keys = jax.random.split(jax.random.key(seed), 3)
    q, k, v = (jax.random.normal(key, SHAPE, jnp.float32) for key in keys)
    return q.astype(jnp.bfloat16), (2 * k).astype(jnp.bfloat16), v.astype(jnp.bfloat16)


def records(query, key, value, mask) -> dict[str, np.ndarray]:
    """The certify launch's own records (`attention_mosaic.certified`)."""
    from attention_witness import certified, spans_of

    out, rows, sums = certified(query, key, value, spans_of(mask, query.shape[0]), interpret=True)
    return {"out": np.asarray(out, np.float64), "m": np.asarray(rows[0], np.float64),
            "l": np.asarray(rows[1], np.float64), "acc": np.asarray(sums, np.float64)}


def refused(found: dict, name: str) -> int:
    rows, inside = found[name]
    return int(np.sum(rows & ~inside))


@pytest.mark.parametrize("causal", [True, False], ids=["causal", "full"])
@pytest.mark.parametrize("seed", [0, 1])
def test_each_stored_state_and_the_output_lie_in_their_step_enclosures(causal, seed):
    q, k, v = operands(seed)
    mask = kept(SHAPE[1], SHAPE[1], causal=causal)
    found = witnessed(q, k, v, mask, interpret=True)
    for name, (rows, inside) in found.items():
        assert rows.any(), name
        assert inside[rows].all(), f"{name}: {refused(found, name)} of {int(rows.sum())} held entries outside"


def test_each_fault_is_refused_at_its_elements():
    q, k, v = operands(2)
    mask = kept(SHAPE[1], SHAPE[1], causal=False)
    truth = records(q, k, v, mask)
    m, acc = truth["m"], truth["acc"]
    alpha = np.exp2(np.where(np.isfinite(m[:, :, :-1]), m[:, :, :-1] - m[:, :, 1:], 0.0))
    dropped = dict(truth, acc=np.concatenate(
        [acc[:, :, :1], acc[:, :, 1:] + acc[:, :, :-1] * (1 - alpha)[..., None]], 2))
    assert refused(witnessed(q, k, v, mask, interpret=True, records=dropped), "acc")
    wrong = dict(truth, m=np.concatenate([m[:, :, :1], m[:, :, :-1]], 2))
    assert refused(witnessed(q, k, v, mask, interpret=True, records=wrong), "m")
    narrowed = dict(truth, acc=np.asarray(acc.astype(jnp.bfloat16), np.float64))
    assert refused(witnessed(q, k, v, mask, interpret=True, records=narrowed), "acc")
    # A causal launch's records held under a mask one key short of its own.
    causal = kept(SHAPE[1], SHAPE[1], causal=True)
    shifted = causal & ~np.eye(SHAPE[1], dtype=bool)[None, None]
    claimed = records(q, k, v, causal)
    held = witnessed(q, k, v, shifted, interpret=True, records=claimed)
    assert sum(refused(held, name) for name in ("m", "l", "acc"))
