"""Every compile of a step, an evaluation or a serving program reads the kernel templates' reports
(`dew.nn.kernels.templates.report.compiled`, row 2384): where a launch's kernel spills, its config is
refused and the program lowered afresh, the launch's derivation taking its next config, and what the
run keeps, its plan, its step compiled ahead and the executable it runs, is the program it lowered
then. A program with no template launch compiles as jit compiles it."""

import os
import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from generation_support import lowers
from jax.experimental.layout import Format, Layout
from process_support import fresh

from dew.inputs import Field, InputSpec
from dew.nn.attention import scaled_dot_product_attention
from dew.nn.kernels import generation, grouped_matmul
from dew.nn.kernels.options import program
from dew.objectives.supervised import Supervised
from dew.telemetry.instrumentation import Compiles
from dew.training import Trainer

ROWS, FEATURES, WIDTH, EXPERTS = 4096, 2048, 4096, 8


class _Uncounted(grouped_matmul.MosaicGmm):
    """The grouped forward with a register model that counts nothing, so its derivation takes a tile its
    registers do not hold: a launch whose model misses, as the kernels' own step-down test makes one."""

    def registers(self, ba, config):
        return 0


class Experts(nn.Module):
    """An MoE layer's paired SwiGLU up projection, each expert taking an equal share of the rows,
    through a launch whose register model misses (`_Uncounted`), and a trained scale over its outputs:
    the launch is the forward kernel alone, so the step differentiates the scale, not the launch. Its
    initialization, traced for the host, draws the weights without launching it. `nonce`, added to the
    rows, makes its step a program no compilation cache holds, whose compile runs ptxas and reports
    what it finds: a cached program reports nothing."""

    nonce: float = 0.0

    @nn.compact
    def __call__(self, x):
        drawn = nn.initializers.normal(0.02)
        gate, up = (jax.lax.stop_gradient(self.param(name, drawn, (EXPERTS, FEATURES, WIDTH), jnp.bfloat16))
                    for name in ("gate", "up"))
        scale = self.param("scale", nn.initializers.ones, (WIDTH,), jnp.float32)
        if self.is_initializing():
            return jnp.zeros((x.shape[0], WIDTH), jnp.bfloat16) * scale
        sizes = jnp.full((EXPERTS,), x.shape[0] // EXPERTS, jnp.int32)
        projected = _Uncounted()(
            jax.lax.stop_gradient((x + self.nonce).astype(jnp.bfloat16)), gate, sizes, paired=up,
            hooks=grouped_matmul.Hooks(epilogue=lambda g, u: jax.nn.silu(g) * u), out_dtype="bfloat16",
            trans_rhs=False, interpret=False)
        return projected * scale


def squared(outputs, batch):
    return jnp.square(outputs.astype(jnp.float32)).mean(axis=-1)


def experts(mesh, nonce: float, ahead=None) -> Trainer:
    return Trainer(Supervised(Experts(nonce), squared, inputs=InputSpec(Field("x", (FEATURES,)))),
                   optax.adam(1e-3), key=0, mesh=mesh, ahead=ahead)


BATCH = {"x": np.zeros((ROWS, FEATURES), np.float32)}


@pytest.fixture
def nonce() -> float:
    """A value no earlier run's step was compiled with (`Experts.nonce`)."""
    return float(time.time_ns() % 10**9) / 10**12


def spilled(tmp_path, script: str, *arguments: str) -> dict:
    """`script`'s last printed JSON, run by a fresh interpreter over a compilation cache of its own,
    apart from every other process's: a process keeps each Mosaic kernel it compiled (jaxlib's
    `KernelHash` cache), and a cache each executable it compiled, either of which runs ptxas again for
    none, so a spill is reported at a kernel's first compile in a process with an empty cache alone."""
    cache = tmp_path / f"cache-{len(list(tmp_path.glob('cache-*')))}"
    return fresh(script, *arguments, cache_dir=cache, flags=os.environ.get("XLA_FLAGS", ""))


AHEAD = """
import json, sys
from test_kernel_spills import BATCH, experts
from dew.nn.kernels.templates import report
from dew.training import MeshSpec
from dew.training.ahead import compile_ahead, compile_only_gpus

nonce, refused = float(sys.argv[1]), sys.argv[3] == "refused"
with report.refusals():
    if refused:
        report.refuse(pickle_load(sys.argv[2]))
    ahead = compile_ahead(experts(MeshSpec().build(compile_only_gpus("sm90", 1)), nonce), BATCH)
    ahead.save(sys.argv[2] + (".again" if refused else ""))
    print(json.dumps({"program": ahead.program, "refusals": len(ahead.refusals),
                      "stepped": [miss.stepped for miss in report.misses()],
                      "missed": [repr(miss.config) for miss in report.misses()],
                      "planned": {name: ahead.plan[name] for name in ("spills", "refused")}}))
"""
"""A step compiled ahead for a compile-only H100, in a registry of its own: the one at argv[2], or, its
argv[3] `refused`, another from the refusals that one carries."""


@lowers("sm90")
def test_a_step_whose_launch_spills_is_compiled_ahead_at_its_next_config(tmp_path, nonce):
    """A step compiled ahead for a compile-only H100 on this CPU, its launch's derived tile spilling at
    ptxas, is refused that tile and lowered once more at the next, the miss recorded (`report.misses`)
    and its plan holding the spill and the refusal, as its telemetry reports them; the step it saves
    carries the refusal (`AheadOfTime.refusals`), and another process that starts from it derives past the
    refused tile, which it never compiles, its plan recording its own refusals alone (train 329's review)."""
    step = str(tmp_path / "step.dew-aot")
    first = spilled(tmp_path, PRELUDE + AHEAD, str(nonce), step, "fresh")
    assert first["refusals"] and first["stepped"][:1] == [True]
    spills, refused = first["planned"]["spills"], first["planned"]["refused"]
    assert [spill["config"] for spill in spills] == first["missed"] and spills[0]["stepped"]
    assert len(refused) == first["refusals"] and refused[0]["config"] == first["missed"][0]
    again = spilled(tmp_path, PRELUDE + AHEAD, str(nonce), step, "refused")
    # Its plan records the refusals its own compiles made, not those it was seeded with.
    assert first["missed"][0] not in again["missed"]
    assert first["missed"][0] not in [refusal["config"] for refusal in again["planned"]["refused"]]


HANDED = """
import json, sys
import jax
from test_kernel_spills import BATCH, experts
from dew._records import listed, record
from dew.nn.kernels.templates import report
from dew.training import MeshSpec
from dew.training.ahead import AheadOfTime

nonce, handed = float(sys.argv[1]), sys.argv[2]
here = MeshSpec().build(jax.devices()[:1])
with report.refusals():
    run = experts(here, nonce, ahead=AheadOfTime.load(handed) if handed else None)
    state = run.fit(BATCH, steps=1, log_every=1)
    first, *_ = listed(record(run.plan, "plan")["probes"], "probes", read=record, what="probes")
    print(json.dumps({"step": int(state.step), "ahead": first["ahead"],
                      "stepped": [miss.stepped for miss in report.misses()]}))
"""
"""A run on this host's GPU, in a registry of its own, handed the step compiled ahead at argv[2], or
none where it is empty."""


SCOPED = """
import json, sys
from dew.nn.kernels.generation import assumed
from dew.nn.kernels.templates import report
from dew.training.ahead import AheadOfTime, handed

step = AheadOfTime.load(sys.argv[1])
identity, config, names = step.refusals[0]
foreign = (identity, config, frozenset({"another kernel"}))
report.refuse([foreign])
with assumed(step.target), handed(step):
    within = report.refused_launches()
print(json.dumps({"within": sorted(map(repr, within)), "step": sorted(map(repr, step.refusals)),
                  "after": repr(foreign) in map(repr, report.refused_launches())}))
"""
"""A process on the step's GPU (`AheadOfTime.target`) that refused a launch of another program, handed the
step compiled ahead at argv[1]."""


@lowers("sm90")
def test_a_run_handed_a_step_compiled_ahead_derives_within_the_steps_refusals_alone(tmp_path, nonce):
    """A process handed a step compiled ahead, whatever launches it or its cache refused before, lowers
    it within the launch configs the step's compiles refused alone (`dew.training.ahead.handed`): one it
    refused of another program is no refusal there, and is the process's own again after (ledger 2453,
    where a refusal shared through a cache led a handed run to derive another program)."""
    step = str(tmp_path / "step.dew-aot")
    spilled(tmp_path, PRELUDE + AHEAD, str(nonce), step, "fresh")
    scoped = spilled(tmp_path, SCOPED, step)
    assert scoped["within"] == scoped["step"] and scoped["after"]


@pytest.mark.gpu(reason="trains a step whose launch spills on this host's H100")
@lowers("sm90")
def test_a_run_steps_its_spilling_launch_down_and_trains(tmp_path, nonce):
    """On an H100, a run whose launch's derived tile spills steps it down once, the miss recorded, and
    trains (row 2384's proof)."""
    trained = spilled(tmp_path, PRELUDE + HANDED, str(nonce), "")
    assert trained["step"] == 1 and trained["stepped"][:1] == [True]


PRELUDE = """
import pickle
def pickle_load(path):
    from dew.training.ahead import AheadOfTime
    return AheadOfTime.load(path).refusals
"""
"""What the scripts share: the refusals a saved step carries."""


@pytest.mark.gpu(reason="hands a step compiled ahead for the canonical H100 to a run on this host's H100")
@lowers("sm90")
def test_a_run_handed_a_step_compiled_ahead_for_another_h100_takes_it(tmp_path, nonce):
    """A run in a fresh process handed a step compiled ahead for the canonical H100, whose launch's first
    tile spilled, on this host's H100, which may count other multiprocessors (an H100 PCIe's 114 against
    132), lowers the saved program: it derives its launches for the target the step was compiled for
    (`AheadOfTime.target`), within the step's refusals, and trains from it, its plan's first compile
    the step compiled ahead (ledger 2453)."""
    step = str(tmp_path / "step.dew-aot")
    spilled(tmp_path, PRELUDE + AHEAD, str(nonce), step, "fresh")
    handed = spilled(tmp_path, PRELUDE + HANDED, str(nonce), step)
    assert handed["step"] == 1 and handed["ahead"] is True


def test_a_handed_step_derives_for_its_target_on_a_gpu_that_runs_it_alone():
    """While a step compiled ahead is reproduced (`handed`), its launches derive for the GPU it was
    compiled for (`AheadOfTime.target`), where this process's GPU is of its generation and holds a
    block's shared memory its launches take, however many multiprocessors it counts; outside, the
    process's own facts. A GPU of another generation, one holding less shared memory a block, or one
    whose shared memory is unknown, is refused (ledger 2453)."""
    import dataclasses

    from dew.nn.kernels.generation import assumed, gpu_target
    from dew.training.ahead import AheadOfTime, handed

    canonical = gpu_target("sm90")
    step = AheadOfTime("sm90", 1, {}, "", b"", (), None, 0, 0, 0, 0, 0, {}, canonical)
    pcie = dataclasses.replace(canonical, device_kind="NVIDIA H100 PCIe", multiprocessors=114)
    with assumed(pcie):
        with handed(step):
            assert gpu_target() == canonical
        assert gpu_target() == pcie
    smaller = dataclasses.replace(pcie, shared_limit=canonical.shared_limit - 1)
    unknown = dataclasses.replace(pcie, shared_limit=None)
    for other in (gpu_target("sm80"), smaller, unknown):
        with assumed(other), pytest.raises(ValueError), handed(step):
            pass


def test_a_program_runs_the_executable_its_calls_kernel_choice_traces():
    """A program (`dew.nn.kernels.options.program`) compiled through the reports keys its executables
    as jit does: called under the fast attention form and the accurate one in turn, it compiles each
    form once and each call computes what jit computes under that form."""
    query = jax.random.normal(jax.random.key(0), (1, 16, 2, 8))

    def attend(query):
        return scaled_dot_product_attention(query, query, query, causal=True)

    run = program(attend)
    ran: dict[str, list[jax.Array]] = {"fast": [], "accurate": []}
    with Compiles("attend") as compiles:
        for form in ("fast", "accurate", "fast", "accurate"):
            with generation.chosen(attention_form=form):
                ran[form].append(run(query))
    assert sum(found.kind == "compiled" for found in compiles.programs) == 2
    for form, outputs in ran.items():
        with generation.chosen(attention_form=form):
            expected = jax.jit(attend)(query)
        for output in outputs:
            np.testing.assert_array_equal(output, expected)


def test_a_programs_calls_of_two_shapes_compile_at_once_from_two_threads():
    """Two threads calling one program at two shapes each lower their own executable while the other
    does, as jit's calls do: each tracing waits for the other's, which a call that held the program for
    its whole compile would never reach (train 329's review: `Decide.compile_scoring`'s threads)."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    met = threading.Barrier(2, timeout=30)

    def doubled(x):
        met.wait()
        return x * 2

    run = program(doubled)
    with ThreadPoolExecutor(2) as pool:
        outputs = list(pool.map(run, (jnp.ones(2), jnp.ones(3))))
    np.testing.assert_array_equal(outputs[1], jnp.full(3, 2.0))


def test_a_program_is_its_jit_within_another_trace_and_keys_its_statics_as_jit_does():
    """Within `jax.grad`, `jax.vmap` or an enclosing jit a program is its jit, traced into theirs, its
    tracers its arguments or ones its function closes over, and under `jax.disable_jit` Python; called
    alone, it keys a static argument by its type as its value, as jit does (True and 1 are two programs),
    and keeps dynamic an argument its jit, given both its static positions and names, takes as one."""
    squared = program(lambda x: x * x)
    assert jax.grad(squared)(3.0) == 6.0
    np.testing.assert_array_equal(jax.vmap(squared)(jnp.arange(3.0)), jnp.arange(3.0) ** 2)
    assert jax.jit(lambda x: squared(x) + 1)(2.0) == 5.0
    assert jax.jit(lambda x: program(lambda: x * x)())(2.0) == 4.0
    with jax.disable_jit():
        assert program(lambda x: x if x > 0 else -x)(-1.0) == 1.0

    def scaled(x, flag):
        return x * 2 if type(flag) is bool else x * 3

    keyed = program(scaled, static_argnums=1)
    named = program(scaled, static_argnums=(), static_argnames="flag")
    x, flag = jnp.ones(2), True
    np.testing.assert_array_equal(keyed(x, flag), x * 2)
    np.testing.assert_array_equal(keyed(x, 1), x * 3)
    np.testing.assert_array_equal(named(x, 2.0), x * 3)
    # An array in another layout is another program's argument, as jit keys it.
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    rows, columns = (jax.device_put(jnp.arange(6.0).reshape(2, 3), Format(Layout(order), sharding))
                     for order in ((0, 1), (1, 0)))
    doubled = program(lambda x: x * 2)
    np.testing.assert_array_equal(doubled(rows), doubled(columns))
    # A typed key array, which states no layout, keys by its sharding.
    drawn = program(lambda key: jax.random.normal(key, (2,)))
    np.testing.assert_array_equal(drawn(jax.random.key(0)), jax.random.normal(jax.random.key(0), (2,)))
