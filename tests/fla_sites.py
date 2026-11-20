"""The gated delta rule's sites the recurrence template is held to fla at: each site's draw, the tests' own
operands and cotangents as a dict of host arrays (`arrays`), which tools/fla_gdn_reference.py hands fla-core
on a GPU and records (tests/fixtures/fla/gdn_seeded.npz), and the rule's hooks.

- `mosaic/<case>/<dtype>/<geometry>`: tests/test_recurrence_mosaic.py's cases (`SHAPES`) in bf16 and in
  fp32 under the fast forms, interpreted ('cpu', `INTERPRETED_LENGTH`) and native ('gpu').
- `walk/<geometry>`: a two-role walk's one sequence, values 64 wide (test_recurrence_mosaic_gradient).
- `near two`: write gates near two on aligned keys (test_recurrence_mosaic).
- `stability/<name>/<regime>/<product>`: test_recurrence_stability's regimes of the delta rule with a decay,
  and of a write gate without it under a decay a head (fla-core's `chunk_simple_gla`, its keys gated), the
  forward's draws and the gradient's (`stressed`, `cotangents`).
- `rwkv6/<dtype>/<layout>`: RWKV6's recurrence (`RWKV6`: a decay a key dimension, read before its update,
  its own write weighed), test_recurrence_chunk's draws of one sequence and of packed documents, against
  fla-core's `chunk_rwkv6`.
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

OWN_LEAVES = ("out", "final", "dq", "dk", "dv", "dg", "down", "dh0")
"""A site's leaves whose readout weighs its own write and reads no write gate (`RWKV6`'s): the own weight's
gradient in the write gate's place."""

RWKV6 = recurrence.Hooks(decay='vector', delta=False, reads='before', own_weight=True, scale=128 ** -0.5)
"""RWKV6's recurrence as fla-core's `chunk_rwkv6` computes it: `o_t = scale q_t^T (S_{t-1} + (u k_t)
v_t^T)`, `S_t = exp(w_t) S_{t-1} + k_t v_t^T` (fla/ops/rwkv6/recurrent_naive.py), its scale fla's default
of keys 128 wide."""

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
                 "beta": operands.beta, "own": operands.own, "h0": operands.state, "dy": out, "dfinal": final,
                 "starts": operands.starts}
        return {name: np.asarray(value) for name, value in named.items() if value is not None}


def chunk_of(arrays: dict) -> tuple[Chunk, tuple]:
    """A site's arrays as the operands and cotangents they were drawn as (`Site.arrays`)."""
    starts = arrays.get("starts")
    return (Chunk(arrays["q"], arrays["k"], arrays["v"], log_decay=arrays["g"], beta=arrays.get("beta"),
                  own=arrays.get("own"), state=arrays["h0"],
                  starts=None if starts is None else jnp.asarray(starts)),
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
                 "dg": grads.log_decay, "dbeta": grads.beta, "down": grads.own, "dh0": grads.state}
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

    sites = {}
    # The interpreted geometry's gradient only where the CPU interprets the case's gradient
    # (`INTERPRETED_GRADIENT`): fla's packed per-key gradient there, which no test reads, resolved no count
    # within 512 seeds at dbeta (fla-unit09-6ac02f3ca).
    interpreted_gradients = {(case, label) for case, label, _ in mosaic.INTERPRETED_GRADIENT}
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


def _rwkv6_sites() -> dict[str, Site]:
    """RWKV6's sites (`RWKV6`): test_recurrence_chunk's draws of one sequence and of packed documents, in bf16
    and fp32, each with its own weight, which the draw gives under `Hooks.own_weight`."""
    import test_recurrence_chunk as chunked

    sites = {}
    for label, dtype in (("bf16", jnp.bfloat16), ("fp32", jnp.float32)):
        for layout, packed in (("one sequence", False), ("packed", True)):
            def draw(seed, dtype=dtype, packed=packed):
                # Values 16 wide: fla-core's rwkv6 backward takes a value block of at least 16 (its dA
                # kernel's `tl.dot(do, v)`, K >= 16).
                (operands,) = chunked.draw(RWKV6, packed, seed, value_width=16)
                return dataclasses.replace(operands, **{name: getattr(operands, name).astype(dtype)
                                                        for name in ("query", "key", "value")})

            sites[f"rwkv6/{label}/{layout}"] = Site(RWKV6, _with_cotangents(draw), OWN_LEAVES)
    return sites


FLOOR_GATES = "kda/tiny gates at the floor/gradient"
"""Kimi delta attention's site of decays at its floor and write gates down to 2^-60 (`_floor_gate_sites`)."""


def _floor_gate_sites() -> dict[str, Site]:
    """Kimi delta attention under unit keys (test_recurrence_stability's 'vector delta', `write_bound`
    declared) with every token's log decay near Kimi Linear's floor ((-5, -4.5) a key dimension) and write
    gates `2^U(-60, 0)`, in 16 bits with their cotangents: a chunk's own pairs and its system's rows into
    fp32's subnormals at once, which no stability regime draws (row 7906)."""
    import test_recurrence_chunk as chunked
    import test_recurrence_stability as stability
    import test_recurrence_template_stability as template_stability

    hooks = stability.SITES['vector delta']

    def gradient(seed):
        operands, (out, final) = stability.cotangents(hooks, 'the decay bound', seed)
        rng = np.random.default_rng([seed, 7906])
        operands = dataclasses.replace(
            operands, log_decay=chunked.hosted(rng.uniform(-5.0, -4.5, operands.log_decay.shape)),
            beta=chunked.hosted(2.0 ** rng.uniform(-60.0, 0.0, operands.beta.shape)))
        return template_stability.bf16(operands, hooks), (out.astype(jnp.bfloat16), final)

    return {FLOOR_GATES: Site(hooks, gradient)}


FAINT_CROSS = "kda/a faint cross term/gradient"
"""Kimi delta attention's site of faint write gates beside large ones over one key (`_faint_cross_sites`)."""


def _faint_cross_sites() -> dict[str, Site]:
    """Kimi delta attention (test_recurrence_stability's 'vector delta') with every token's key and value the
    unit `e0`, no decay, write gates alternating faint (2^-U(20, 40)) and large (U[1/2, 1]), the queries
    `e0` but each faint token's zero, in 16 bits with their cotangents: each large token's own pair beside
    the faint cross term the token before left, which its log decay's gradient takes once its own pair is
    summed apart (row 7906; its cell compared one gradient bitwise outside the gate, row 8569)."""
    import test_recurrence_chunk as chunked
    import test_recurrence_stability as stability
    import test_recurrence_template_stability as template_stability

    hooks = stability.SITES['vector delta']

    def gradient(seed):
        operands, (out, final) = stability.cotangents(hooks, 'the decay bound', seed)
        rng = np.random.default_rng([seed, 8569])
        unit = np.zeros(operands.key.shape)
        unit[..., 0] = 1.0
        faint = np.arange(operands.beta.shape[1])[None, :, None] % 2 == 0
        beta = np.where(faint, 2.0 ** -rng.uniform(20.0, 40.0, operands.beta.shape),
                        rng.uniform(0.5, 1.0, operands.beta.shape))
        query = np.where(faint[..., None], 0.0, unit)
        value = np.zeros(operands.value.shape)
        value[..., 0] = 1.0
        operands = dataclasses.replace(
            operands, query=chunked.hosted(query), key=chunked.hosted(unit), value=chunked.hosted(value),
            log_decay=chunked.hosted(np.zeros(operands.log_decay.shape)), beta=chunked.hosted(beta))
        return template_stability.bf16(operands, hooks), (out.astype(jnp.bfloat16), final)

    return {FAINT_CROSS: Site(hooks, gradient)}


@functools.cache
def sites() -> dict[str, Site]:
    """Every recorded site by name."""
    return {**_mosaic_sites(), **_stability_sites(), **_rwkv6_sites(), **_floor_gate_sites(),
            **_faint_cross_sites()}


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
