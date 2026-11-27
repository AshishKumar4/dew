"""Mamba2's rated scan's sites the recurrence template is held to mamba_ssm at (row 6781): the stability
suite's regimes of the rated scan (test_recurrence_stability's `SITES['rated']`), its forward's draws and its
gradient's as test_recurrence_template_stability takes them, in bf16; each site's draw as host arrays
(`Site.arrays`), which tools/mamba_stability_reference.py hands mamba_ssm's own `mamba_chunk_scan_combined`
on a GPU and records (`RECORDED`), and Dew's leaves of it (`ruled`).

- `stability/rated/<regime>/forward`: the forward's draws (`stressed`), its output and final memory.
- `stability/rated/<regime>/gradient`: the gradient's (`cotangents`, its documents restarted where the
  regime resets them), its output, final memory and every operand's gradient under the drawn cotangents.

A recorded site draws the same inputs on every host (testing.md, `input_digest`): every normal of a site's
draw, its transform's and its cotangents' are `recorded_inputs.normals`' (`_normals`, keyed by the regime's
name, the product and the seed), its uniforms NumPy's
integer-derived ones, its arithmetic float64 on the host rounded once (`hosted`).
"""

import dataclasses
import functools
import itertools
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from dew.nn.kernels.recurrence_chunk import Chunk

RECORDED = Path(__file__).resolve().parent / "fixtures" / "mamba2" / "stability_seeded.npz"
"""mamba_ssm's recording at every site (tools/mamba_stability_reference.py)."""

OPERANDS = ("value", "beta", "rate", "key", "query", "skip", "state")
"""The rated scan's operands mamba_ssm takes (`ssd_support.MAMBA_SIDE`'s leaves, `Chunk`'s names): its values
`x`, its step `dt`, its rate `A`, its keys `B`, its queries `C`, its skip `D` and the memory entering."""

FORWARD = ("out", "final")
"""A forward's site's leaves."""

LEAVES = (*FORWARD, *(f"d_{name}" for name in OPERANDS))
"""A gradient's site's leaves: the output and final memory, then each operand's gradient (`MAMBA_SIDE`'s
names)."""


class Site(NamedTuple):
    draw: Callable[[int], tuple[Chunk, tuple[jax.Array, jax.Array]]]
    """A seed's operands (their documents' starts in place) and the cotangents of the output and final
    memory."""
    leaves: tuple[str, ...] = LEAVES

    def arrays(self, seed: int) -> dict[str, np.ndarray]:
        """The seed's draw as host arrays, the recording's inputs."""
        operands, (out, final) = self.draw(seed)
        named = {**{name: getattr(operands, name) for name in OPERANDS}, "dy": out, "dfinal": final,
                 "starts": operands.starts}
        return {name: np.asarray(value) for name, value in named.items() if value is not None}


def chunk_of(arrays: dict) -> tuple[Chunk, tuple]:
    """A site's arrays as the operands and cotangents they were drawn as (`Site.arrays`)."""
    starts = arrays.get("starts")
    return (Chunk(arrays["query"], arrays["key"], arrays["value"], state=arrays["state"],
                  starts=None if starts is None else jnp.asarray(starts), beta=arrays["beta"],
                  rate=arrays["rate"], skip=arrays["skip"]),
            (arrays["dy"], arrays["dfinal"]))


def ruled(rule: Callable[[Chunk], tuple], arrays: dict, leaves: tuple[str, ...] = LEAVES) -> dict:
    """`rule`'s leaves on a site's arrays (those of `leaves`): its output and final memory, and where asked,
    each operand's gradient under the drawn cotangents, its documents' starts held fixed."""
    operands, (dy, dfinal) = chunk_of(arrays)
    starts = operands.starts

    def run(given: Chunk):
        return rule(dataclasses.replace(given, starts=starts))

    given = dataclasses.replace(operands, starts=None)
    if set(leaves) <= set(FORWARD):
        out, final = run(given)
        return {name: value for name, value in (("out", out), ("final", final)) if name in leaves}
    (out, final), pull = jax.vjp(run, given)
    (grads,) = pull((dy.astype(out.dtype), dfinal.astype(final.dtype)))
    found = {"out": out, "final": final, **{f"d_{name}": getattr(grads, name) for name in OPERANDS}}
    return {name: found[name] for name in leaves}


def _normals(*tags) -> Callable[[tuple], np.ndarray]:
    """A draw's normals in turn, float64, each `recorded_inputs.normals`' of `tags` and its place in the
    draw: the same bits on every host."""
    from recorded_inputs import normals

    places = itertools.count()
    return lambda shape: normals([6781, *tags, next(places)], shape, np.float64)


def sites() -> dict[str, Site]:
    """Every recorded site by name: each regime's forward and gradient of the rated scan."""
    import test_recurrence_stability as stability
    from test_recurrence_chunk import cotangents as drawn_cotangents
    from test_recurrence_template_stability import bf16

    hooks = stability.SITES["rated"]
    found = {}
    for regime in stability.REGIMES:
        # A regime's streams keyed by its name, so another regime added or moved leaves its draws' digests.
        tag = zlib.crc32(regime.encode())

        def forward(seed, tag=tag, regime=regime):
            normal = _normals(tag, 0, seed)
            operands = bf16(stability.stressed(hooks, regime, seed, normal=normal)[0], hooks)
            return operands, drawn_cotangents(operands, seed, normal=normal)

        def gradient(seed, tag=tag, regime=regime):
            normal = _normals(tag, 1, seed)
            operands, (out, final) = stability.cotangents(hooks, regime, seed, normal=normal)
            restored = stability.restarting(lambda given: given, regime)(operands)
            return bf16(restored, hooks), (out.astype(jnp.bfloat16), final)

        found[f"stability/rated/{regime}/forward"] = Site(forward, FORWARD)
        found[f"stability/rated/{regime}/gradient"] = Site(gradient)
    return found


def the_template_is_as_exact_as_mamba_ssm(rule: Callable[[Chunk], tuple], site: str, label: str | None = None,
                                          under: Callable = lambda run: run) -> None:
    """`rule` (a chunked call, `chunk_on`'s) on par from float64 with recorded mamba_ssm at `site`'s declared
    seeds (check 1), its leaves the site's, the truth the recurrence a token at a time in float64, which the
    recorder held to mamba_ssm's at every seed; `under` wrapping Dew's arm, which check 1 compiles."""
    import test_recurrence_stability as stability
    from reference_error import as_given, assert_as_exact, declared_seeds, indexed_recording, recording
    from test_recurrence_chunk import token_by_token

    found = sites()[site]
    seeds, leaves = declared_seeds(RECORDED)[site], found.leaves
    draw, reference, truth = indexed_recording(
        recording(RECORDED), site, seeds, inputs_of=found.arrays, leaves=leaves,
        truth_fn=jax.jit(lambda arrays: ruled(functools.partial(token_by_token, stability.SITES["rated"]),
                                              arrays, leaves)))

    def run(index, arrays):
        del index
        return ruled(rule, arrays, leaves)

    assert_as_exact(under(run), as_given(reference), draw, label or f"the template against mamba_ssm, {site}",
                    seeds=seeds, truth_fn=as_given(truth))
