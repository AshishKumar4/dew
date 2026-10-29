"""Every compile of a step, an evaluation or a serving program reads the kernel templates' reports
(`dew.nn.kernels.templates.report.compiled`, row 2384): where a launch's kernel spills, its config is
refused and the program lowered afresh, the launch's derivation taking its next config, and what the
run keeps, its plan, its step compiled ahead and the executable it runs, is the program it lowered
then. A program with no template launch compiles as jit compiles it."""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from generation_support import lowers

from dew.inputs import Field, InputSpec
from dew.nn.attention import scaled_dot_product_attention
from dew.nn.kernels import generation, grouped_matmul
from dew.nn.kernels.options import program
from dew.nn.kernels.templates import report
from dew.objectives.supervised import Supervised
from dew.telemetry.instrumentation import Compiles
from dew.training import MeshSpec, Trainer

ROWS, FEATURES, WIDTH, EXPERTS = 4096, 2048, 4096, 8


class _Uncounted(grouped_matmul.MosaicGmm):
    """The grouped forward with a register model that counts nothing, so its derivation takes a tile its
    registers do not hold: a launch whose model misses, as the kernels' own step-down test makes one."""

    def registers(self, ba, config):
        return 0


class Experts(nn.Module):
    """An MoE layer's paired SwiGLU up projection, each expert taking an equal share of the rows,
    through a launch whose register model misses (`_Uncounted`), and a trained scale over its outputs:
    the launch is the forward kernel alone, so the step differentiates the scale, not the launch."""

    @nn.compact
    def __call__(self, x):
        drawn = nn.initializers.normal(0.02)
        gate, up = (jax.lax.stop_gradient(self.param(name, drawn, (EXPERTS, FEATURES, WIDTH), jnp.bfloat16))
                    for name in ("gate", "up"))
        sizes = jnp.full((EXPERTS,), x.shape[0] // EXPERTS, jnp.int32)
        projected = _Uncounted()(
            jax.lax.stop_gradient(x.astype(jnp.bfloat16)), gate, sizes, paired=up,
            hooks=grouped_matmul.Hooks(epilogue=lambda g, u: jax.nn.silu(g) * u), out_dtype="bfloat16",
            trans_rhs=False, interpret=False)
        return projected * self.param("scale", nn.initializers.ones, (WIDTH,), jnp.float32)


def squared(outputs, batch):
    return jnp.square(outputs.astype(jnp.float32)).mean(axis=-1)


def experts(mesh, ahead=None) -> Trainer:
    return Trainer(Supervised(Experts(), squared, inputs=InputSpec(Field("x", (FEATURES,)))),
                   optax.adam(1e-3), key=0, mesh=mesh, ahead=ahead)


BATCH = {"x": np.zeros((ROWS, FEATURES), np.float32)}


@lowers("sm90")
def test_a_step_whose_launch_spills_is_compiled_ahead_at_its_next_config():
    """A step compiled ahead for a compile-only H100 on this CPU, its launch's derived tile spilling at
    ptxas, is refused that tile and lowered once more at the next, the miss recorded (`report.misses`);
    the step it saves is that program and carries the refusal (`AheadOfTime.refusals`): a process that
    starts from the refusal lowers that program at once, with nothing to step down."""
    from dew.training.ahead import compile_ahead, compile_only_gpus

    def compiled():
        return compile_ahead(experts(MeshSpec().build(compile_only_gpus("sm90", 1))), BATCH)

    with report.refusals():
        first = compiled()
        stepped = [miss.stepped for miss in report.misses()]
    assert first.refusals and stepped[:1] == [True]
    with report.refusals():
        report.refuse(first.refusals)
        again = compiled()
        assert not any(miss.stepped for miss in report.misses())
    assert again.program == first.program


@pytest.mark.gpu(reason="trains a step whose launch spills, and one compiled ahead, on this host's H100")
@lowers("sm90")
def test_a_run_steps_its_spilling_launch_down_and_one_handed_the_step_compiled_ahead_takes_it():
    """On an H100, a run whose launch's derived tile spills steps it down once, the miss recorded, and
    trains (row 2384's proof); a run in a fresh process's registry handed that step compiled ahead
    refuses the compile's tile before it lowers (`FitSearch`), so it lowers the saved program and
    trains from it, its plan's first compile the step compiled ahead."""
    from dew._records import listed, record
    from dew.training.ahead import compile_ahead, compile_only_gpus

    here = MeshSpec().build(jax.devices()[:1])
    with report.refusals():
        state = experts(here).fit(BATCH, steps=1, log_every=1)
        assert [miss.stepped for miss in report.misses()][:1] == [True]
    assert int(state.step) == 1
    with report.refusals():
        ahead = compile_ahead(experts(MeshSpec().build(compile_only_gpus("sm90", 1))), BATCH)
    with report.refusals():
        run = experts(here, ahead=ahead)
        state = run.fit(BATCH, steps=1, log_every=1)
    first, *_ = listed(record(run.plan, "plan")["probes"], "probes", read=record, what="probes")
    assert int(state.step) == 1 and first["ahead"] is True


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
