"""The recurrence's check-1 matrix (tests/kernel_matrix.py): the chunked recurrence's training step on par
with the upstream's, through each kernel its routes name.

Its cells run generation.CHOICES' `gated_delta_rule` and `ssd`. Each takes the chunked op
(`recurrence_chunk.chunk`) of a form, forward and backward in one program as a layer's training step takes
them, with the kernel the cell names (`generation.chosen`):
- the recurrence template ('mosaic_gpu');
- the portable math ('xla', the template's plain-JAX mainloop).

Its output, final memory and every operand's gradient are held to its family's independent oracle at the
cell's dtype (check 1; docs/design/testing.md, kernel-templates.md 9), the recurrence a token at a time in
float64 the truth: the upstream's own code where the family has a release (fla's for the delta rule's forms,
mamba_ssm's for the rated scan, `RELEASES`), live in
the GPU tier's reference interpreters (`DEW_FLA_PYTHON`'s, `DEW_MAMBA_PYTHON`'s), else the recurrence a token
at a time. Dew's own chunk math is no cell's reference: a cell's distance from it is reported beside the
verdict, with the forms the call took. What a kernel serves is the route's own: the step traced under the
cell's generation, its selectors deciding (`recurrence_chunk.port_kernel`, `scan_kernel`), a named kernel
insisting on the call and refusing what it does not take (`KernelRefused`).
"""

import contextlib
import dataclasses
import functools
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from delta_rule_support import fla_layer_side
from gpu_flow import cell
from kernel_matrix import AXES
from reference_error import (
    FEWEST_SEEDS,
    assert_as_exact,
    indexed_recording,
    recording,
    widened,
    with_gradients,
)
from ssd_support import mamba_side
from test_recurrence_chunk import FORMED, VALUE_HEADS, chunk_on, draw, hosted, token_by_token
from upstream_support import upstream_arm

from dew.nn.kernels import generation, recurrence_chunk
from dew.nn.kernels.selection import KernelRefused
from dew.parity import MOST_SEEDS
from dew.parity_stages import REDRAW

FORMS = {"gated delta net": FORMED[0], "kimi delta attention": FORMED[2], "write gate": FORMED[1],
         "rated scan": FORMED[4]}
"""Each `form` axis value's hooks (test_recurrence_chunk's): Qwen3.5's gated delta net (its gates and l2
norm), Kimi Linear's decay a key dimension with its output norm, a step write gate without the delta rule,
and Mamba-2's rated scan with its skip."""

OPS = {op.choice: op for op in (recurrence_chunk.GATED_DELTA, recurrence_chunk.SCAN)}
"""The chunked op's selectors by their choice's name, each declaring its kernels."""

CHOICES = frozenset(OPS)
"""The generation.CHOICES entries the family's cells run (row 184): the chunked op's selectors'."""

KERNELS = {choice: frozenset(kernel.name for kernel in op.kernels) for choice, op in OPS.items()}
"""The kernels each choice's cells run under test (row 4513), its selector's every kernel, each taken by its
route (the cell asserts it): a kernel added to an op is the matrix's without a second edit."""

FLA_LAYERS = {"gated delta net": "gdn", "kimi delta attention": "kda"}
"""fla's own layer each delta rule form is (`delta_rule_support.FLA_LAYER_SIDE`): Qwen3.5's gated delta net
and Kimi Linear's."""

RELEASES = {**dict.fromkeys(FLA_LAYERS, "fla"), "rated scan": "mamba_ssm"}
"""The upstream library each form's family has a release of (testing.md: a cell's oracle is that release's
own code): fla's layers for the delta rule's forms, mamba_ssm's scan for Mamba-2's rated scan; the step
write gate's form has none."""


def _fla(named: dict[str, str], leaves: dict, cotangents: tuple, starts) -> dict:
    """fla's own layer of `named`'s form on stacked draws (`FLA_LAYERS`, `fla_layer_side`)."""
    hooks = FORMS[named["form"]]
    return fla_layer_side(FLA_LAYERS[named["form"]], leaves, cotangents, starts, jnp.dtype(named["dtype"]),
                          epsilon=0.0 if hooks.output is None else hooks.output.epsilon)


def _mamba(named: dict[str, str], leaves: dict, cotangents: tuple, starts) -> dict:
    """mamba_ssm's own scan on stacked draws (`mamba_side`), in the chunks the cell's call takes."""
    del named
    return mamba_side(leaves, cotangents, starts, recurrence_chunk.CHUNK_SIZE)


UPSTREAMS = {"fla": _fla, "mamba_ssm": _mamba}
"""Each upstream library a form's family has a release of (`RELEASES`), run on a cell's stacked draws in its
GPU tier reference interpreter."""

RECORDED = {"fla": Path(__file__).resolve().parent / "fixtures" / "fla" / "matrix_seeded.npz"}
"""Each upstream's recording of the cells a host without its GPU tier reference judges (a CPU lane's
interpreted cells; tools/fla_matrix_reference.py), the same upstream at its pinned release as a live run's."""

INTERPRETERS = {"fla": "DEW_FLA_PYTHON", "mamba_ssm": "DEW_MAMBA_PYTHON"}
"""The variable naming each upstream's GPU tier reference interpreter, where a host holds one."""

UPSTREAM_SEEDS = 8 * MOST_SEEDS
"""The most of a cell's seeds its upstream runs (`_upstream`): every seed a sizing reads (`seeds_for`'s
reference against itself reads four times its count, from offsets past it)."""



def _op(hooks) -> generation.Choice:
    """The choice a form's call takes its kernel from: the rated scan's (`recurrence_chunk.SCAN`) or the
    delta rule's (`GATED_DELTA`)."""
    op = recurrence_chunk.SCAN if hooks.rated and not hooks.delta else recurrence_chunk.GATED_DELTA
    return op.choice




FP32_HEADS = 16
"""The value heads an fp32 cell draws (`_drawn`): its per-head parameters' gradients (a gate's `log_a` and
`bias`, a rate) an element a head, which at the draw's two left three fp32 cells' paired ratios unresolved
within 512 seeds (M-JS-75d); a 16-bit cell's rounding resolves at two."""


def _drawn(case: dict[str, str], seed: int):
    """`case`'s operands from `seed` (`draw`, its queries, keys and values and write gate in the case's
    dtype; an fp32 cell's over `FP32_HEADS` value heads) and fp32 cotangents of its output and final
    memory."""
    hooks = FORMS[case["form"]]
    heads = FP32_HEADS if case["dtype"] == "float32" else VALUE_HEADS
    (operands,) = draw(hooks, packed=case["documents"] == "packed", seed=seed, value_heads=heads)
    narrow = jnp.dtype(case["dtype"])
    operands = dataclasses.replace(operands, starts=None, **{
        name: getattr(operands, name).astype(narrow) for name in ("query", "key", "value", "beta")
        if getattr(operands, name) is not None})
    batch, length, heads, width = operands.value.shape
    # Host NumPy's, as the operands are (`hosted`): a recording's draws digest alike on any host.
    normal = np.random.default_rng([seed, 101]).standard_normal
    return operands, (hosted(normal((batch, length, heads, width))),
                      hosted(normal((batch, heads, operands.key.shape[-1], width))))


def _inputs(case: dict[str, str], seed: int) -> dict[str, np.ndarray]:
    """`case`'s draw at `seed` (`_drawn`) as a recording reads it: its operands' arrays by name (a static
    field, `Chunk.held`, no operand) and its cotangents (`d_out`, `d_final`), host arrays."""
    operands, (d_out, d_final) = _drawn(case, seed)
    arrays = jax.tree_util.tree_leaves_with_path(operands)
    return {**{path[0].name: np.asarray(value) for path, value in arrays},
            "d_out": np.asarray(d_out), "d_final": np.asarray(d_final)}


def _split(case: dict[str, str], inputs: dict) -> tuple:
    """A recording's draw (`_inputs`) as `case`'s step takes it: its operands and cotangents."""
    operands = dataclasses.replace(_drawn(case, 0)[0], **{name: jnp.asarray(value) for name, value in
                                                          inputs.items() if name not in ("d_out", "d_final")})
    return operands, (jnp.asarray(inputs["d_out"]), jnp.asarray(inputs["d_final"]))


def _leaves(results) -> dict:
    """A step's results (`_step`'s output, final memory and operand gradients) by the names a recording keeps
    them under: `out`, `final` and `d_<operand>`."""
    (out, final), grads = results
    return {"out": out, "final": final, **{f"d_{path[0].name}": value for path, value in
                                            jax.tree_util.tree_leaves_with_path(grads)}}


def _starts(case: dict[str, str]):
    """`case`'s documents' first tokens (`draw`'s, the same for every seed), None for one sequence."""
    if case["documents"] != "packed":
        return None
    (operands,) = draw(FORMS[case["form"]], packed=True, seed=0)
    return operands.starts


def _step(case: dict[str, str], kernel: str):
    """`case`'s training step through the chunked op on `kernel`: the output and final memory, and every
    operand's gradient under their cotangents (`with_gradients`)."""
    hooks, starts = FORMS[case["form"]], _starts(case)

    def ours(operands):
        return _wide(chunk_on(kernel, hooks, dataclasses.replace(operands, starts=starts)))

    return with_gradients(ours)


def _wide(results):
    """A call's output and final memory in at least fp32, as its cotangents are drawn (float64 in the truth,
    which widens them): the cast's pullback rounds them to the call's own dtypes."""
    return jax.tree.map(lambda x: x.astype(jnp.promote_types(x.dtype, jnp.float32)), results)


@functools.cache
def _routed(case: tuple[tuple[str, str], ...], kernel: str) -> frozenset | None:
    """The routes (`generation.routes`: choice and kernel) `case`'s training step on `kernel` takes under its
    generation, traced (`jax.eval_shape`), the route's own selectors deciding; None where a named kernel
    refuses it (`KernelRefused`)."""
    named = dict(case)
    traced = False
    # A named kernel refusing the call serves it not: no routes.
    with generation.assumed(named["generation"]), generation.routes() as found, \
            contextlib.suppress(KernelRefused):
        jax.eval_shape(_step(named, kernel), *_drawn(named, 0))
        traced = True
    return frozenset((route.choice, route.kernel) for route in found) if traced else None


def _reference(case: dict[str, str]) -> str:
    """`case`'s family's independent oracle at its dtype: the upstream's own code (`upstream <library>`,
    `UPSTREAMS`) where its form's family has a release (`RELEASES`), whatever kernel the cell names and
    whether Dew's port of it takes the call, else the recurrence a token at a time ('token': a step write
    gate's form, which no release runs). Held instead to the token oracle wherever Dew's port refused a call,
    a cell of a released family would have dropped its upstream with no rule saying so (row 6150)."""
    release = RELEASES.get(case["form"])
    return "token" if release is None else f"upstream {release}"


def supported(case: dict[str, str]) -> bool:
    """Whether the route serves `case` (tests/kernel_matrix.py): its kernel takes the case's training call
    (`_routed`, the route's own selectors: not refused, nor handed to another kernel); a kernel of another
    op's (`KERNELS`) is none of its form's."""
    if case["kernel"] not in KERNELS[_op(FORMS[case["form"]])]:
        return False
    taken = _routed(tuple(case.items()), case["kernel"])
    return taken is not None and any(kernel == case["kernel"] for _, kernel in taken)


CASES = (
    ('sm90', 'gated delta net', 'mosaic_gpu', 'bfloat16', 'one'),
    ('sm90', 'rated scan', 'pallas', 'bfloat16', 'packed'),
    ('sm80', 'gated delta net', 'fla', 'bfloat16', 'packed'),
    ('sm80', 'rated scan', 'pallas', 'bfloat16', 'one'),
    ('sm80', 'rated scan', 'mosaic_gpu', 'bfloat16', 'one'),
    ('sm90', 'kimi delta attention', 'xla', 'float32', 'one'),
    ('sm80', 'write gate', 'xla', 'float32', 'packed'),
    ('sm80', 'kimi delta attention', 'mosaic_gpu', 'bfloat16', 'packed'),
    ('sm90', 'write gate', 'xla', 'bfloat16', 'one'),
    ('sm90', 'kimi delta attention', 'fla', 'bfloat16', 'one'),
    ('sm90', 'gated delta net', 'xla', 'float32', 'one'),
    ('sm90', 'rated scan', 'xla', 'float32', 'one'),
)
"""The cells' cases, each its axis values in `AXES`' order: every pair of axis values the route serves
(`kernel_matrix.covering("recurrence")`, regenerated where the axes or the route's support change), under
the suite's lane environment: at its 'highest' an fp32 training call is served by the portable math alone,
the reference."""

SEEDS = {
    ('sm90', 'gated delta net', 'mosaic_gpu', 'bfloat16', 'one'): 256,
    ('sm90', 'rated scan', 'pallas', 'bfloat16', 'packed'): 8,
    ('sm80', 'gated delta net', 'fla', 'bfloat16', 'packed'): 64,
    ('sm80', 'rated scan', 'pallas', 'bfloat16', 'one'): 8,
    ('sm80', 'rated scan', 'mosaic_gpu', 'bfloat16', 'one'): 64,
    ('sm90', 'kimi delta attention', 'xla', 'float32', 'one'): 128,
    ('sm80', 'write gate', 'xla', 'float32', 'packed'): 64,
    ('sm80', 'kimi delta attention', 'mosaic_gpu', 'bfloat16', 'packed'): 8,
    ('sm90', 'write gate', 'xla', 'bfloat16', 'one'): 64,
    ('sm90', 'kimi delta attention', 'fla', 'bfloat16', 'one'): 8,
    ('sm90', 'gated delta net', 'xla', 'float32', 'one'): 256,
    ('sm90', 'rated scan', 'xla', 'float32', 'one'): 512,
}
"""Each cell's declared seeds, sized by `seeds_for` on its own GPU, paired against its oracle (`_reference`,
the upstream run live or the token recurrence; an H100 and an A100, M-JS-75d at b44df0fea0e, and the fp32
cells over `FP32_HEADS` on an A100 and a GH200, M-JS-77 and M-JS-77b at 89af07e0eda); a cell the covering
took for its red row at fdb49a44d0a its form, kernel and dtype's sized cell's until its own GPU sizes it."""

TPU_CASES = (
    ('v6e', 'rated scan', 'tpu', 'bfloat16', 'one'),
    ('v5e', 'rated scan', 'tpu', 'float32', 'packed'),
    ('v6e', 'rated scan', 'tpu', 'bfloat16', 'packed'),
    ('v6e', 'rated scan', 'tpu', 'float32', 'one'),
    ('v5e', 'rated scan', 'tpu', 'bfloat16', 'one'),
)
"""The TPU cells' cases (`kernel_matrix.covering("recurrence_tpu")`)."""


def _upstream(named: dict[str, str], block: int, *, first: int = 0):
    """The upstream's own computation of `named`'s cell at its dtype (`upstream_support.upstream_arm`): its
    form's release's (`RELEASES`, `UPSTREAMS`) on the cell's stacked draws, read back a draw at a time, the
    same tree as the cell's own step, up to `UPSTREAM_SEEDS` past `first`."""
    run, starts = UPSTREAMS[RELEASES[named["form"]]], _starts(named)

    def run_block(drawn):
        # The draw's arrays alone: a static field (`Chunk.held`) is no operand of the upstream's.
        names = [path[0].name for path, _ in jax.tree_util.tree_leaves_with_path(drawn[0][0])]
        found = run(named, {name: np.stack([getattr(operands, name) for operands, _ in drawn])
                            for name in names},
                    tuple(np.stack(parts) for parts in zip(*(c for _, c in drawn), strict=True)), starts)
        return [((jnp.asarray(found["out"][index]), jnp.asarray(found["final"][index])),
                 dataclasses.replace(operands, **{name: jnp.asarray(found[f"d_{name}"][index],
                                                                    getattr(operands, name).dtype)
                                                  for name in names}))
                for index, (operands, _) in enumerate(drawn)]

    return upstream_arm(run_block, functools.partial(_drawn, named), block, most=UPSTREAM_SEEDS, first=first,
                        keyed=lambda operands, cotangents: operands)


def _deficit(named: dict[str, str], seeds: int, truth) -> str:
    """The cell's kernel's distance from float64 against Dew's own chunk math's (the portable math's), each
    leaf's pooled RMS ratio over the first `seeds` draws: reported beside the verdict, never judged."""
    port, portable = (jax.jit(_step(named, kernel)) for kernel in (named["kernel"], "xla"))
    errors: dict[str, dict[str, list[float]]] = {}
    for seed in range(seeds):
        drawn = _drawn(named, seed)
        with jax.enable_x64(new_val=True):
            exact = truth(*widened(drawn))
        for arm, results in (("port", port(*drawn)), ("portable", portable(*drawn))):
            for (path, leaf), true in zip(jax.tree_util.tree_flatten_with_path(results)[0],
                                         jax.tree.leaves(exact), strict=True):
                errors.setdefault(jax.tree_util.keystr(path), {}).setdefault(arm, []).append(
                    float(np.mean((np.asarray(leaf, np.float64) - np.asarray(true, np.float64)) ** 2)))
    return ", ".join(f"{name} {np.sqrt(sum(arms['port']) / max(sum(arms['portable']), 1e-300)):.3g}"
                     for name, arms in errors.items())


def _on_par(family: str, case: tuple[str, ...], seeds: int = FEWEST_SEEDS) -> None:
    """The chunked op's training step of `case` on its GPU, its kernel named, on par with its reference's
    (`_reference`; check 1, the recurrence a token at a time in float64 the truth) over `seeds` draws: the
    output, the final memory and every operand's gradient. Its routes are its family's (`CHOICES`). A cell
    reports its distance from Dew's own chunk math and the forms its call took."""
    named = dict(zip(AXES[family], case, strict=True))
    hooks, starts = FORMS[named["form"]], _starts(named)

    def truth(operands):
        return _wide(token_by_token(hooks, dataclasses.replace(operands, starts=starts)))

    step = jax.jit(_step(named, named["kernel"]))
    # Its routes read from one draw's step: a check-1 site's judge pools its map tasks' records, running none.
    with generation.routes() as found:
        jax.block_until_ready(step(*_drawn(named, 0)))
    assert {route.choice for route in found} <= CHOICES
    assert any(route.kernel == named["kernel"] for route in found)
    reference = _reference(named)
    upstream = reference.removeprefix("upstream ") if reference.startswith("upstream ") else None
    if upstream is not None and not os.environ.get(INTERPRETERS[upstream]):
        _on_par_recorded(named, step, with_gradients(truth), upstream, seeds)
        return
    oracle = jax.jit(with_gradients(truth)) if upstream is None else _upstream(named, (1 + REDRAW) * seeds)
    assert_as_exact(step, oracle, functools.partial(_drawn, named), "-".join(case), seeds=seeds,
                    truth_fn=with_gradients(truth))
    if named["kernel"] != "xla":
        print(f"{'-'.join(case)} under the {generation.numerics().forms} forms, its distance against Dew's "
              f"own chunk math's: {_deficit(named, FEWEST_SEEDS, with_gradients(truth))}", flush=True)


def _on_par_recorded(named: dict[str, str], step, truth, upstream: str, seeds: int) -> None:
    """`_on_par` of a cell on a host without its upstream's GPU tier reference, over its declared `seeds`:
    held to the upstream's recording of it (`RECORDED`, a summary over at least their staged draws, which
    `indexed_recording` refuses short of), its draws drawn again on this host (`_inputs`, NumPy's) and Dew's
    float64 truth held to the recorded truth's sketch; a recording of another release than the tier's pin
    is stale, refused."""
    from tool_support import load

    site = "-".join(named.values())
    # Only the recorded upstreams' cells are held here: another's is its GPU tier's alone.
    assert upstream in RECORDED, (f"{site}: no recording of {upstream}; its cells run where "
                                  f"{INTERPRETERS[upstream]} names its reference interpreter")
    stored = recording(RECORDED[upstream])
    recorded = json.loads(str(stored["provenance"]))["revision"]
    pinned = load("reference_releases").RELEASES[upstream][f"{upstream}-core"]
    assert recorded == pinned, f"{site}: recorded on {upstream} {recorded}, the tier pins {pinned}: stale"
    drawn, reference, recorded_truth = indexed_recording(
        stored, site, seeds, inputs_of=functools.partial(_inputs, named),
        truth_fn=lambda inputs: _leaves(truth(*_split(named, inputs))))
    assert_as_exact(lambda index, inputs: _leaves(step(*_split(named, inputs))), reference, drawn, site,
                    seeds=seeds, truth_fn=recorded_truth)


@cell("recurrence", covers=lambda case: dict(zip(AXES["recurrence"], case, strict=True)))
@pytest.mark.parametrize("case", [pytest.param(case, marks=pytest.mark.skipif(
    generation.device_generation() != case[0], reason=f"an {case[0]} cell runs on its own GPU alone"))
    for case in CASES], ids="-".join)
def a_recurrences_training_step_is_on_par(case: tuple[str, ...]) -> None:
    """`_on_par` of a GPU's case at its declared seeds (`SEEDS`)."""
    _on_par("recurrence", case, SEEDS[case])


@cell("recurrence_tpu", covers=lambda case: dict(zip(AXES["recurrence_tpu"], case, strict=True)))
@pytest.mark.parametrize("case", [pytest.param(case, marks=pytest.mark.skipif(
    generation.device_generation() != case[0], reason=f"an {case[0]} cell runs on its own GPU alone"))
    for case in TPU_CASES], ids="-".join)
def a_recurrences_training_step_on_a_tpu_is_on_par(case: tuple[str, ...]) -> None:
    """`_on_par` of a TPU's case (the TPU tier's flow, tests/test_ssd_kernel.py)."""
    _on_par("recurrence_tpu", case)
