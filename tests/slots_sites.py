"""The slot compositions' sites they are held to fla at (`dew.nn.kernels.slots`, row 6127): each site's draw
as host arrays (`Site.arrays`), which tools/fla_slots_reference.py hands fla-core's `chunk_lightnet` and
`chunk_gsa` on a GPU and records (`RECORDED`), and Dew's leaves of it (`ruled`).

- `lightnet/<dtype>`: LightNet's recurrence from an entering memory, no normalizer entering.
- `gsa/<dtype>`: gated slot attention, its slots' log decays given, from entering memories of both passes.
- `abc/<dtype>`: ABC's, its slots' decays their own running normalizer, from entering memories: fla's
  chunk_gsa with no gates (`g=None`), its differentiable additive gate's.

Each over two rows of 80 tokens (a chunk of 64 and a part one) and two heads, keys 32 wide, 16 slots, values
16 wide, in fp32 and bf16 (the decays, memories and the final memories' cotangents fp32).
"""

import functools
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from dew.nn.kernels import slots

RECORDED = Path(__file__).resolve().parent / "fixtures" / "fla" / "slots_seeded.npz"
"""fla-core's recording at every site (tools/fla_slots_reference.py)."""

ROWS, LENGTH, HEADS, KEYS, SLOTS, VALUES = 2, 80, 2, 32, 16, 16

LEAVES = {
    "lightnet": ("out", "final", "dq", "dk", "dv", "dh0"),
    "gsa": ("out", "hk", "hv", "dq", "dk", "dv", "ds", "dg", "dhk0", "dhv0"),
    "abc": ("out", "hk", "hv", "dq", "dk", "dv", "ds", "dhk0", "dhv0"),
}
"""Each form's leaves: its output, final memories and each operand's gradient under the drawn cotangents."""


class Site(NamedTuple):
    form: str
    draw: Callable[[int], dict]
    """A seed's operands and the cotangents of the output and the final memories, by name."""

    @property
    def leaves(self) -> tuple[str, ...]:
        return LEAVES[self.form]

    def arrays(self, seed: int) -> dict[str, np.ndarray]:
        """The seed's draw as host arrays, the recording's inputs."""
        return {name: np.asarray(value) for name, value in self.draw(seed).items()}


def _drawn(seed: int, *, form: str, dtype) -> dict:
    """A seed's operands (queries and keys of unit scale, values and slots' writes normal, in `dtype`; GSA's
    slots' log decays `-(n^2 + 1/8) / 8`, IEEE arithmetic of a normal `n`, about `log sigmoid(n + 2)`'s
    mean; entering memories fp32) and its cotangents: the same bits on every host, as a recorded site
    draws them (testing.md, `input_digest`), normals by `recorded_inputs.normals` and no libm."""
    from recorded_inputs import normals
    from test_recurrence_chunk import hosted

    draws = iter(range(12))

    def normal(*shape, scale=1.0, kind=jnp.float32):
        """The seed's next normals in float64, scaled, rounded once to fp32 and then to `kind`."""
        return hosted(normals([6127, seed, tuple(LEAVES).index(form), next(draws)], shape, np.float64)
                      * scale, kind)

    q, k = (normal(ROWS, LENGTH, HEADS, KEYS, scale=KEYS ** -0.5, kind=dtype) for _ in range(2))
    drawn = {"q": q, "k": k, "v": normal(ROWS, LENGTH, HEADS, VALUES, kind=dtype),
             "dy": normal(ROWS, LENGTH, HEADS, VALUES, kind=dtype)}
    if form == "lightnet":
        return drawn | {"h0": normal(ROWS, HEADS, KEYS, VALUES), "dfinal": normal(ROWS, HEADS, KEYS, VALUES)}
    drawn |= {"s": normal(ROWS, LENGTH, HEADS, SLOTS, kind=dtype),
              "hk0": normal(ROWS, HEADS, KEYS, SLOTS), "hv0": normal(ROWS, HEADS, SLOTS, VALUES),
              "dhk": normal(ROWS, HEADS, KEYS, SLOTS), "dhv": normal(ROWS, HEADS, SLOTS, VALUES)}
    if form == "gsa":
        n = normals([6127, seed, tuple(LEAVES).index(form), next(draws)], (ROWS, LENGTH, HEADS, SLOTS),
                    np.float64)
        drawn["g"] = hosted(-(n * n + 0.125) / 8)
    return drawn


def ruled(rule: Callable, form: str, arrays: dict) -> dict:
    """`rule` (`slots.lightnet` or `slots.slot_attention`'s signature) on a site's arrays: its leaves under
    the drawn cotangents (`LEAVES[form]`)."""
    given = {name: jnp.asarray(value) for name, value in arrays.items()}
    if form == "lightnet":
        def run(q, k, v, h0):
            out, (final, _) = rule(q, k, v, state=h0)
            return out, final

        (out, final), pull = jax.vjp(run, *(given[name] for name in ("q", "k", "v", "h0")))
        grads = pull((given["dy"].astype(out.dtype), given["dfinal"]))
        return {"out": out, "final": final} | dict(zip(("dq", "dk", "dv", "dh0"), grads, strict=True))
    names = ("q", "k", "v", "s", "hk0", "hv0") + (("g",) if form == "gsa" else ())

    def attend(q, k, v, s, hk0, hv0, g=None):
        out, finals = rule(q, k, v, s, g, state=(hk0, hv0))
        return out, tuple(finals)[:2]

    (out, (hk, hv)), pull = jax.vjp(attend, *(given[name] for name in names))
    grads = pull((given["dy"].astype(out.dtype), (given["dhk"], given["dhv"])))
    found = {"out": out, "hk": hk, "hv": hv}
    found |= dict(zip(("dq", "dk", "dv", "ds", "dhk0", "dhv0", "dg"), grads, strict=False))
    return {name: found[name] for name in LEAVES[form]}


def by_token(form: str) -> Callable:
    """The form's recurrences a token at a time (tests/test_slots.py's), `slots`' signatures."""
    from test_slots import lightnet_by_token, slots_by_token

    if form == "lightnet":
        def lightnet(q, k, v, state=None):
            out, final = lightnet_by_token(q, k, v, state)
            return out, (final, None)

        return lightnet

    def attend(q, k, v, s, g=None, state=None):
        return slots_by_token(q, k, v, s, *state, g)

    return attend


def sites() -> dict[str, Site]:
    """Every recorded site by name."""
    return {f"{form}/{name}": Site(form, functools.partial(_drawn, form=form, dtype=dtype))
            for form in LEAVES for name, dtype in (("fp32", jnp.float32), ("bf16", jnp.bfloat16))}


def the_slots_are_as_exact_as_fla(site: str) -> None:
    """`slots`' form on par from float64 with recorded fla-core at `site`'s declared seeds (check 1), the
    truth its recurrences a token at a time in float64, which the recorder held to fla's own reference at
    every seed."""
    from reference_error import as_given, assert_as_exact, declared_seeds, indexed_recording, recording

    form = sites()[site].form
    rule = slots.lightnet if form == "lightnet" else slots.slot_attention
    seeds = declared_seeds(RECORDED)[site]
    draw, reference, truth = indexed_recording(
        recording(RECORDED), site, seeds, inputs_of=sites()[site].arrays, leaves=LEAVES[form],
        truth_fn=jax.jit(lambda arrays: ruled(by_token(form), form, arrays)))

    def run(index, arrays):
        del index
        return ruled(rule, form, arrays)

    assert_as_exact(run, as_given(reference), draw, f"{site} against fla", seeds=seeds,
                    truth_fn=as_given(truth))
