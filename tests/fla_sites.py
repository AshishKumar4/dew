"""The gated delta rule's sites the recurrence template is held to fla at: each site's draw, the tests' own
operands and cotangents as a dict of host arrays (`arrays`), which tools/fla_gdn_reference.py hands fla-core
on a GPU and records (tests/fixtures/fla/gdn_seeded.npz), and the rule's hooks.

- `mosaic/<case>/<dtype>/<geometry>`: tests/test_recurrence_mosaic.py's cases (`SHAPES`) in bf16 and in
  fp32 under the fast forms, interpreted ('cpu', `INTERPRETED_LENGTH`) and native ('gpu').
- `walk/<geometry>`: a two-role walk's one sequence, values 64 wide (test_recurrence_mosaic_gradient).
- `near two`: write gates near two on aligned keys (test_recurrence_mosaic).
- `stability/<name>/<regime>/<product>`: test_recurrence_stability's regimes of the delta rule with a decay,
  the forward's draws and the gradient's (`stressed`, `cotangents`).
"""

import dataclasses
import functools
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from dew.nn.kernels import recurrence
from dew.nn.kernels.recurrence_chunk import Chunk

RECORDED = Path(__file__).resolve().parent / "fixtures" / "fla" / "gdn_seeded.npz"
"""fla-core's recording at every site (tools/fla_gdn_reference.py)."""

LEAVES = ("out", "final", "dq", "dk", "dv", "dg", "dbeta", "dh0")
"""A recorded site's leaves: the output and final memory, then each operand's gradient."""

FORWARD = ("out", "final")
"""A forward's site's leaves."""


class Site(NamedTuple):
    hooks: recurrence.Hooks
    draw: Callable[[int], tuple[Chunk, tuple[jax.Array, jax.Array]]]
    """A seed's operands (their documents' starts in place) and the cotangents of the output and final
    memory."""
    leaves: tuple[str, ...] = LEAVES
    """The leaves its tests read, which the recorder records: a forward's site its output and final memory
    alone, no gradient run."""
    paired: bool = False
    """Whether its recording sizes its seed count paired with the template's own arm (`record_case`'s
    `dew`), as check 1's verdict pairs them: a site whose errors are heavy-tailed across draws alike in both
    arms, where fla compared with itself on other draws resolved no count within 512 seeds (the final
    memory under decays near zero, fla-unit14-6ac02f3ca, fla-unit15-6ac02f3ca)."""

    def arrays(self, seed: int) -> dict[str, np.ndarray]:
        """The seed's draw as host arrays, the recording's inputs."""
        operands, (out, final) = self.draw(seed)
        named = {"q": operands.query, "k": operands.key, "v": operands.value, "g": operands.log_decay,
                 "beta": operands.beta, "h0": operands.state, "dy": out, "dfinal": final}
        if operands.starts is not None:
            named["starts"] = operands.starts
        return {name: np.asarray(value) for name, value in named.items()}


def chunk_of(arrays: dict) -> tuple[Chunk, tuple]:
    """A site's arrays as the operands and cotangents they were drawn as (`Site.arrays`)."""
    starts = arrays.get("starts")
    return (Chunk(arrays["q"], arrays["k"], arrays["v"], log_decay=arrays["g"], beta=arrays["beta"],
                  state=arrays["h0"], starts=None if starts is None else jnp.asarray(starts)),
            (arrays["dy"], arrays["dfinal"]))


def ruled(rule: Callable[[Chunk], tuple], arrays: dict, leaves: tuple[str, ...] = LEAVES) -> dict:
    """`rule`'s leaves on a site's arrays (`LEAVES`, those of `leaves`): its output and final memory, and
    where asked, each operand's gradient under the drawn cotangents, its documents' starts held fixed."""
    operands, cotangents = chunk_of(arrays)
    starts = operands.starts

    def run(given: Chunk):
        return rule(dataclasses.replace(given, starts=starts))

    given = dataclasses.replace(operands, starts=None)
    if not set(leaves) - {"out", "final"}:
        out, final = run(given)
        found = {"out": out, "final": final}
    else:
        (out, final), pull = jax.vjp(run, given)
        (grads,) = pull((cotangents[0].astype(out.dtype), cotangents[1].astype(final.dtype)))
        found = {"out": out, "final": final, "dq": grads.query, "dk": grads.key, "dv": grads.value,
                 "dg": grads.log_decay, "dbeta": grads.beta, "dh0": grads.state}
    return {name: found[name] for name in leaves}


def _with_cotangents(draw: Callable[[int], Chunk]) -> Callable[[int], tuple[Chunk, tuple]]:
    """A draw of operands with cotangents of their output, in its dtype, and their final memory, fp32,
    from the seed's own stream (`test_recurrence_chunk.cotangents`, test_recurrence_mosaic_gradient's
    `drawn`'s)."""
    from test_recurrence_chunk import cotangents

    def drawn(seed: int):
        operands = draw(seed)
        return operands, cotangents(operands, seed)

    return drawn


def _mosaic_sites() -> dict[str, Site]:
    import test_recurrence_mosaic as mosaic
    import test_recurrence_mosaic_gradient as gradient

    sites = {}
    # The interpreted geometry's gradient only where the CPU interprets the case's gradient
    # (`INTERPRETED_GRADIENT`): fla's packed per-key gradient there, which no test reads, resolved no count
    # within 512 seeds at dbeta (fla-unit09-6ac02f3ca).
    interpreted_gradients = {(case, label) for case, label, _ in gradient.INTERPRETED_GRADIENT}
    for case in mosaic.SHAPES:
        hooks = mosaic.case_hooks(case)
        for label, dtype in mosaic.DTYPES.items():
            for geometry, interpreted in (('cpu', True), ('gpu', False)):
                at = mosaic.geometry_on(case, interpreted)
                leaves = FORWARD if interpreted and (case, label) not in interpreted_gradients else LEAVES
                sites[f"mosaic/{case}/{label}/{geometry}"] = Site(hooks, _with_cotangents(
                    lambda seed, dtype=dtype, at=at: mosaic.draw(dtype, seed, **at)[0]), leaves)
    for geometry, interpreted in (('cpu', True), ('gpu', False)):
        at = {**mosaic.geometry_on('one sequence', interpreted), 'value_width': 64}
        hooks = recurrence.Hooks(decay='scalar', scale=at['key_width'] ** -0.5, write_bound=1.0)
        sites[f"walk/{geometry}"] = Site(hooks, _with_cotangents(
            lambda seed, at=at: mosaic.draw(jnp.bfloat16, seed, **at)[0]))
    sites["near two"] = Site(mosaic.NEAR_TWO_HOOKS, _with_cotangents(mosaic.near_two), FORWARD)
    return sites


def _stability_sites() -> dict[str, Site]:
    import test_recurrence_stability as stability
    import test_recurrence_template_stability as template_stability

    sites = {}
    for name, hooks in stability.SITES.items():
        if not template_stability.fla_recorded(hooks):
            continue
        for regime in stability.REGIMES:
            def forward(seed, hooks=hooks, regime=regime):
                return template_stability.bf16(stability.stressed(hooks, regime, seed)[0], hooks)

            def gradient(seed, hooks=hooks, regime=regime):
                operands, (out, final) = stability.cotangents(hooks, regime, seed)
                restored = stability.restarting(lambda given: given, regime)(operands)
                return template_stability.bf16(restored, hooks), (out.astype(jnp.bfloat16), final)

            sites[f"stability/{name}/{regime}/forward"] = Site(hooks, _with_cotangents(forward), FORWARD,
                                                               paired=regime == "decays near zero")
            sites[f"stability/{name}/{regime}/gradient"] = Site(hooks, gradient)
    return sites


@functools.cache
def sites() -> dict[str, Site]:
    """Every recorded site by name."""
    return {**_mosaic_sites(), **_stability_sites()}


def the_template_is_as_exact_as_fla(rule: Callable[[Chunk], tuple], site: str,
                                    leaves: tuple[str, ...] = LEAVES, label: str | None = None,
                                    under: Callable = lambda run: run) -> None:
    """`rule` (a chunked call, `chunk_on`'s) on par from float64 with recorded fla-core at `site`'s declared
    seeds (check 1), its leaves those of `leaves`, the truth the recurrence a token at a time in float64,
    which the recorder held to fla's own reference at every seed; `under` wrapping Dew's arm, which check 1
    compiles."""
    from reference_error import as_given, assert_as_exact, declared_seeds, indexed_recording, recording
    from test_recurrence_chunk import token_by_token

    seeds, hooks = declared_seeds(RECORDED)[site], sites()[site].hooks
    draw, reference, truth = indexed_recording(
        recording(RECORDED), site, seeds, inputs_of=sites()[site].arrays, leaves=leaves,
        truth_fn=jax.jit(lambda arrays: ruled(functools.partial(token_by_token, hooks), arrays, leaves)))

    def run(index, arrays):
        del index
        return ruled(rule, arrays, leaves)

    assert_as_exact(under(run), as_given(reference), draw, label or f"the template against fla, {site}",
                    seeds=seeds, truth_fn=as_given(truth))
