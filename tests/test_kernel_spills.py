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
    """`script`'s last printed JSON, run by a fresh interpreter over a compilation cache of its own: a
    process keeps each Mosaic kernel it compiled (jaxlib's `KernelHash` cache), which runs ptxas again
    for none, so a spill is reported at a kernel's first compile in a process alone."""
    return fresh(script, *arguments, cache_dir=tmp_path / "cache", flags=os.environ.get("XLA_FLAGS", ""))


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
                      "missed": [repr(miss.config) for miss in report.misses()]}))
"""
"""A step compiled ahead for a compile-only H100, in a registry of its own: the one at argv[2], or, its
argv[3] `refused`, another from the refusals that one carries."""


@lowers("sm90")
def test_a_step_whose_launch_spills_is_compiled_ahead_at_its_next_config(tmp_path, nonce):
    """A step compiled ahead for a compile-only H100 on this CPU, its launch's derived tile spilling at
    ptxas, is refused that tile and lowered once more at the next, the miss recorded (`report.misses`);
    the step it saves carries the refusal (`AheadOfTime.refusals`), and another process that starts from
    it derives past the refused tile, which it never compiles."""
    step = str(tmp_path / "step.dew-aot")
    first = spilled(tmp_path, PRELUDE + AHEAD, str(nonce), step, "fresh")
    assert first["refusals"] and first["stepped"][:1] == [True]
    again = spilled(tmp_path, PRELUDE + AHEAD, str(nonce), step, "refused")
    assert first["missed"][0] not in again["missed"]


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


@pytest.mark.gpu(reason="trains a step whose launch spills, and one compiled ahead, on this host's H100")
@lowers("sm90")
def test_a_run_steps_its_spilling_launch_down_and_one_handed_the_step_compiled_ahead_takes_it(tmp_path,
                                                                                             nonce):
    """On an H100, a run whose launch's derived tile spills steps it down once, the miss recorded, and
    trains (row 2384's proof); a run in a fresh process handed that step compiled ahead refuses the
    compile's tile before it lowers (`FitSearch`), so it lowers the saved program and trains from it,
    its plan's first compile the step compiled ahead."""
    trained = spilled(tmp_path, PRELUDE + HANDED, str(nonce), "")
    assert trained["step"] == 1 and trained["stepped"][:1] == [True]
    step = str(tmp_path / "step.dew-aot")
    spilled(tmp_path, PRELUDE + AHEAD, str(-nonce), step, "fresh")
    handed = spilled(tmp_path, PRELUDE + HANDED, str(-nonce), step)
    assert handed["step"] == 1 and handed["ahead"] is True and not any(handed["stepped"])


PRELUDE = """
import pickle
def pickle_load(path):
    from dew.training.ahead import AheadOfTime
    return AheadOfTime.load(path).refusals
"""
"""What the scripts share: the refusals a saved step carries."""


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
