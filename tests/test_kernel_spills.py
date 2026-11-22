"""Every compile of a step, an evaluation or a serving program reads the kernel templates' reports
(`dew.nn.kernels.templates.report.compiled`, row 2384): where a launch's kernel spills, its config is
refused and the program lowered afresh, the launch's derivation taking its next config, and what the
run keeps, its plan, its step compiled ahead and the executable it runs, is the program it lowered
then. A program with no template launch compiles as jit compiles it."""

import collections
import dataclasses
import functools
import gc
import os
import subprocess
import time
import weakref

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from lane_environment import DETERMINISTIC_OPS
from generation_support import lowers
from jax.experimental.layout import Format, Layout
from kernel_compile import Uncounted
from process_support import fresh, keeping_picks

from dew.inputs import Field, InputSpec
from dew.nn.attention import scaled_dot_product_attention
from dew.nn.kernels import generation, grouped_matmul
from dew.nn.kernels.options import Fetched, compiled_with, owned_program, program
from dew.nn.kernels.templates import assembled, report, sass
from dew.nn.kernels.templates.gemm.hooks import Hooks
from dew.nn.precision import Numerics
from dew.objectives.supervised import Supervised
from dew.telemetry.compiles import Compiles
from dew.training import Trainer

ROWS, FEATURES, WIDTH, EXPERTS = 4096, 2048, 4096, 8


class Experts(nn.Module):
    """An MoE layer's paired SwiGLU up projection, each expert taking an equal share of the rows,
    through a launch whose register model misses (`kernel_compile.Uncounted`), and a trained scale over its
    outputs: the launch is the forward kernel alone, so the step differentiates the scale, not the launch. Its
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
        projected = Uncounted()(
            jax.lax.stop_gradient((x + self.nonce).astype(jnp.bfloat16)), gate, sizes, paired=up,
            hooks=Hooks(epilogue=lambda g, u: jax.nn.silu(g) * u), out_dtype="bfloat16",
            trans_rhs=False, interpret=False)
        return projected * scale


def squared(outputs, batch):
    return jnp.square(outputs.astype(jnp.float32)).mean(axis=-1)


def experts(mesh, nonce: float, ahead=None) -> Trainer:
    return Trainer(Supervised(Experts(nonce), squared, inputs=InputSpec(Field("x", (FEATURES,)))),
                   optax.adam(1e-3), key=0, mesh=mesh, ahead=ahead)


BATCH = {"x": np.zeros((ROWS, FEATURES), np.float32)}


@pytest.fixture(name="nonce")
def fresh_compilation_nonce() -> float:
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
import dataclasses, json, sys
from test_kernel_spills import BATCH, experts
from dew.nn.kernels.templates import report
from dew.training import MeshSpec
from dew.nn.kernels.generation import compile_only_gpus
from dew.training.ahead import AheadOfTime, compile_ahead

nonce, refused = float(sys.argv[1]), sys.argv[3] == "refused"
with report.refusals():
    if refused:
        report.refuse(pickle_load(sys.argv[2]))
    # An H100's 80 GiB, which a step is compiled ahead against (`compile_only_gpus`' memory).
    ahead = compile_ahead(experts(MeshSpec().build(compile_only_gpus("sm90", 1, 80 * 2**30)), nonce), BATCH)
    ahead.save(sys.argv[2] + (".again" if refused else ""))
    # The saved plan read back beside every field of each miss the compile made, its config by its text.
    misses = [{**{field.name: getattr(miss, field.name) for field in dataclasses.fields(miss)},
               "config": repr(miss.config)} for miss in report.misses()]
    read = AheadOfTime.load(sys.argv[2] + (".again" if refused else "")).plan["spills"]
    print(json.dumps({"program": ahead.program, "refusals": len(ahead.refusals), "misses": misses,
                      "read": read,
                      "stepped": [miss.stepped for miss in report.misses()],
                      "missed": [repr(miss.config) for miss in report.misses()],
                      "planned": {name: ahead.plan[name] for name in ("spills", "refused")}}))
"""
"""A step compiled ahead for a compile-only H100, in a registry of its own: the one at argv[2], or, its
argv[3] `refused`, another from the refusals that one carries."""


@lowers("sm90")
def test_a_step_whose_launch_spills_is_compiled_ahead_at_its_next_config(tmp_path, nonce):
    """A step compiled ahead for a compile-only H100 on this CPU, its launch's derived tile spilling at
    ptxas, is refused that tile and lowered again at the next until one spills nothing, the miss recorded
    (`report.misses`) and its plan holding the spill and the refusal, as its telemetry reports them; the
    step it saves
    carries the refusal (`AheadOfTime.refusals`), and another process that starts from it derives past the
    refused tile, which it never compiles, its plan recording its own refusals alone (train 329's review).
    The saved plan read back holds every field of each miss (row 6518: a serialization-only refusal's codes
    were dropped)."""
    step = str(tmp_path / "step.dew-aot")
    first = spilled(tmp_path, PRELUDE + AHEAD, str(nonce), step, "fresh")
    assert first["refusals"] and first["stepped"][:1] == [True]
    spills, refused = first["planned"]["spills"], first["planned"]["refused"]
    assert [spill["config"] for spill in spills] == first["missed"] and spills[0]["stepped"]
    # The saved plan, read back, holds every field of each miss the compile made (row 6518).
    assert first["read"] == first["misses"]
    assert len(refused) == first["refusals"] and refused[0]["config"] == first["missed"][0]
    again = spilled(tmp_path, PRELUDE + AHEAD, str(nonce), step, "refused")
    # Its plan records the refusals its own compiles made, not those it was seeded with.
    assert first["missed"][0] not in again["missed"]
    assert first["missed"][0] not in [refusal["config"] for refusal in again["planned"]["refused"]]


RECURRENT_AHEAD = """
import json
import sys
import jax
import optax
from token_support import token_rows
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.kernels.generation import compile_only_gpus
from dew.nn.kernels.templates import report
from dew.nn.mixers.gated_delta_net import GatedDeltaNetMixer
from dew.objectives.lm import LMObjective
from dew.training import MeshSpec, Trainer
from dew.training.ahead import compile_ahead

# Each program lowered, by name, and each Mosaic GPU kernel Pallas lowered afresh, by its module's text.
import hashlib
from jax._src.pallas.mosaic_gpu import lowering as pallas_lowering
events, modules = [], []
jax.monitoring.register_event_duration_secs_listener(
    lambda event, seconds, *, fun_name=None, **_: events.append(fun_name)
    if event == "/jax/core/compile/jaxpr_to_mlir_module_duration" else None)
afresh = pallas_lowering.lower_unpipelined_jaxpr_to_module


def lowered_afresh(*args, **kwargs):
    lowered = afresh(*args, **kwargs)
    modules.append(hashlib.sha256(str(lowered.module).encode()).hexdigest())
    return lowered


pallas_lowering.lower_unpipelined_jaxpr_to_module = lowered_afresh
# Qwen3.5-4B's linear attention: 16 key heads and 32 value heads, each 128 wide, over a row of 12,288 tokens.
model = CausalTransformer(vocab_size=64, emb_features=512, num_layers=1, num_heads=4, head_dim=128,
                          mlp_features=512, max_seq_len=12288, dtype="bfloat16", mixer=GatedDeltaNetMixer())
devices = compile_only_gpus(sys.argv[1], 1, 80 * 2**30)
trainer = Trainer(LMObjective(model, 12288), optax.adam(1e-3), key=jax.random.key(0),
                  mesh=MeshSpec().build(devices))
compile_ahead(trainer, token_rows(64, 1, 12289))
print(json.dumps({"events": events, "modules": modules, "kind": devices[0].device_kind,
                  "stored": [certification.stored for certification in report.certifications()],
                  "misses": [[miss.kernel, miss.op, repr(miss.config), miss.stepped]
                             for miss in report.misses()]}))
"""
"""A one-layer decoder of Qwen3.5-4B's gated delta net compiled ahead for a compile-only GPU of the generation
or the kind argv[1] names (`compile_only_gpus`): each program it lowered by name, each Mosaic GPU kernel
lowered afresh by its module's text, the kind its device reported, and whether each launch it certified
read its verdict back from the store (`report.certifications`' `stored`)."""


@pytest.mark.parametrize("target", [pytest.param("sm90", marks=lowers("sm90")),
                                    pytest.param("sm100", marks=lowers("sm100"))])
@pytest.mark.parametrize("deterministic", [False, True], ids=["keeping picks", "deterministic ops"])
def test_a_recurrent_step_is_lowered_once_cold_and_once_warm(tmp_path, target, deterministic):
    """A one-layer decoder of the 4B's gated delta net, its step compiled ahead for a compile-only H100 and a
    compile-only B200 (`RECURRENT_AHEAD`), twice over one cache: each process lowers the step once, the cold
    one certifies each launch's configs once each (whatever its walk takes: a config refused steps the
    launch's producer down at trace, each rung compiled alone with its outputs live, as a step reads them),
    and the warm one reads the verdicts stored, certifying nothing. Certification that dropped the outputs
    never compiled the output launch, and the program's compile stepped it down a whole-step relowering a
    rung, again in every process (LF-PROMO3: 8 lowerings of the 4B's step, cold and warm). Each
    certification's program is named by its launch, direction and config
    (`certified_<op>_l<launch>_c<config>[_reverse]`), so a start's events tell each apart. Each kernel is
    lowered afresh once in the cold process: the step's kernels certification lowered are the lowerings it
    made (a launch's trace held, `tuning.lowered_once`, bound into the step as certification traced it,
    under the step's own mesh), where the step lowered its kernels again, each a cache miss, 34 of a 4B's 36
    kernels on a B200 the ones certification had made (FS-CERTSPLIT4: 27 s of its lowering). Under
    deterministic ops too, which keep no picks but keep the verdicts (row 7308: every warm start under
    --xla_gpu_deterministic_ops certified every launch again)."""
    cache = tmp_path / "cache"
    # The lane's own flags each time, without deterministic ops and with them, whatever the lane's are.
    flags = f"{keeping_picks()} {DETERMINISTIC_OPS}" if deterministic else keeping_picks()
    runs = [fresh(RECURRENT_AHEAD, target, cache_dir=cache, flags=flags) for _ in range(2)]
    cold, warm = (run["events"] for run in runs)
    modules = runs[0]["modules"]
    assert len(set(modules)) == len(modules), collections.Counter(modules).most_common(3)
    assert [run.count("jit(step)") for run in (cold, warm)] == [1, 1], (cold, warm)
    certified = [name for name in cold if name.startswith("jit(certified_")]
    assert certified and len(set(certified)) == len(certified), certified
    assert not [name for name in warm if name.startswith("jit(certified_")], warm
    # Each launch certification refused, within the settle that refused it, is recorded once a refused config
    # and stepped: the retry certified it past the rung (`report.settling`), as a B200's reverse launch at its
    # first was (FS-ASIDE-SM100C).
    misses = runs[0]["misses"]
    assert all(stepped for *_, stepped in misses), misses
    assert len({tuple(miss) for miss in misses}) == len(misses), misses


@lowers("sm90")
def test_a_verdict_certified_for_a_gpu_is_read_by_another_kind_of_its_generation(tmp_path):
    """Certification's verdicts are the compile target's (`report.verdict_target`): a step compiled ahead for
    a compile-only H100 certifies its launches and stores their verdicts, and a process certifying the same
    launches on another kind of sm90 (a compile-only GH200) reads them, certifying
    nothing (rows 6717, 7322: a GH200's run read none of a compile-only H100's verdicts and certified each
    launch again)."""
    cache = tmp_path / "cache"
    flags = keeping_picks()
    h100 = fresh(RECURRENT_AHEAD, "sm90", cache_dir=cache, flags=flags)
    gh200 = fresh(RECURRENT_AHEAD, "NVIDIA GH200 480GB", cache_dir=cache, flags=flags)
    assert h100["kind"] != gh200["kind"] == "NVIDIA GH200 480GB", (h100["kind"], gh200["kind"])
    # Each launch certified (`report.certifications`): the H100's compiled there, the GH200's read back.
    assert h100["stored"] and not any(h100["stored"]), h100["stored"]
    assert gh200["stored"] and all(gh200["stored"]), gh200["stored"]


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


def a_cached_spilling_run_trains_its_stepped_program(tmp_path, nonce):
    """A restart over the same cache reads the cached first program's kernel binaries, finds the spill
    again and steps down to the stepped program, not the refused one. Different compiler flags compile and
    judge their own program, without inherited refusals."""
    shared, flags = tmp_path / "cache", os.environ.get("XLA_FLAGS", "")
    first = fresh(PRELUDE + HANDED, str(nonce), "", cache_dir=shared, flags=flags)
    again = fresh(PRELUDE + HANDED, str(nonce), "", cache_dir=shared, flags=flags)
    assert first["step"] == again["step"] == 1
    assert first["stepped"][:1] == again["stepped"][:1] == [True]
    other = fresh(PRELUDE + HANDED, str(nonce), "", cache_dir=shared,
                  flags=f"{flags} --xla_gpu_enable_latency_hiding_scheduler=true")
    assert other["step"] == 1 and other["stepped"][:1] == [True]


SCOPED = """
import json, sys
from dew.nn.kernels.generation import assumed
from dew.nn.kernels.templates import report
from dew.training.ahead import AheadOfTime, handed

step = AheadOfTime.load(sys.argv[1])
identity, config, names = step.refusals[0]
foreign = (identity, config, frozenset({("another kernel", None)}))
report.refuse([foreign])
with assumed(step.target), handed(step):
    within = report.refused_launches()
print(json.dumps({"within": sorted(map(repr, within)), "step": sorted(map(repr, step.refusals)),
                  "after": repr(foreign) in map(repr, report.refusal_entries())}))
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
def test_a_run_handed_a_step_compiled_ahead_for_the_canonical_h100_takes_it_on_one_of_its_multiprocessors(
        tmp_path, nonce):
    """A run in a fresh process handed a step compiled ahead for the canonical H100, whose launch's first
    tile spilled, on this host's H100 lowers the saved program where this H100 counts its 132
    multiprocessors: it derives its launches for the target the step was compiled for
    (`AheadOfTime.target`), within the step's refusals, and trains from it, its plan's first compile the
    step compiled ahead (ledger 2453). Where it counts others (an H100 PCIe's 114) the run is refused, as a
    balanced tail's grid derived past a GPU's residency waits on a program that never starts (row 3606)."""
    step = str(tmp_path / "step.dew-aot")
    spilled(tmp_path, PRELUDE + AHEAD, str(nonce), step, "fresh")
    canonical = generation.gpu_target("sm90").multiprocessors
    if generation.device_target(jax.devices()[0]).multiprocessors != canonical:
        from dew.training.ahead import AheadOfTime, handed

        with pytest.raises(ValueError), handed(AheadOfTime.load(step)):
            pass
        return
    handed = spilled(tmp_path, PRELUDE + HANDED, str(nonce), step)
    assert handed["step"] == 1 and handed["ahead"] is True


def test_a_handed_step_derives_for_its_target_on_a_gpu_that_runs_it_alone():
    """While a step compiled ahead is reproduced (`handed`), its launches derive for the GPU it was
    compiled for (`AheadOfTime.target`), where this process's GPU is of its generation, counts as many
    multiprocessors and holds a block's shared memory its launches take, whatever its name or memory;
    outside, the process's own facts. A GPU of another generation, of other multiprocessors (an H100
    PCIe's 114: a balanced tail's grid derived past a GPU's residency waits on a program that never
    starts, row 3606), holding less shared memory a block, or whose shared memory is unknown, is refused
    (ledger 2453)."""
    import dataclasses

    from dew.nn.kernels.generation import assumed, gpu_target
    from dew.training.ahead import AheadOfTime, Executable, handed

    canonical = gpu_target("sm90")
    step = AheadOfTime("sm90", 1, {}, "", Executable(b"", ()), None, 0, 0, 0, 0, 0, {},
                       target=canonical)
    h200 = dataclasses.replace(canonical, device_kind="NVIDIA H200")
    with assumed(h200):
        with handed(step):
            assert gpu_target() == canonical
        assert gpu_target() == h200
    # A step compiled ahead before steps recorded their GPU, unpickled with no target, is refused so too.
    earlier = AheadOfTime("sm90", 1, {}, "", Executable(b"", ()), None, 0, 0, 0, 0, 0, {})
    vars(earlier).pop("target")
    with assumed(h200), pytest.raises(ValueError), handed(earlier):
        pass
    pcie = dataclasses.replace(canonical, device_kind="NVIDIA H100 PCIe", multiprocessors=114)
    smaller = dataclasses.replace(h200, shared_limit=canonical.shared_limit - 1)
    unknown = dataclasses.replace(h200, shared_limit=None)
    for other in (gpu_target("sm80"), pcie, smaller, unknown):
        with assumed(other), pytest.raises(ValueError), handed(step):
            pass


def test_a_program_runs_the_executable_its_calls_kernel_choice_traces():
    """A program (`dew.nn.kernels.options.program`) compiled through the reports keys its executables
    as jit does: called under a run's fast attention forms and its accurate ones in turn
    (`generation.run_precision`), it compiles each form on its first call (a launch that spills there
    compiled again a step down, as on a GH200) and nothing on its second, and each call computes what jit
    computes under that form."""
    query = jax.random.normal(jax.random.key(0), (1, 16, 2, 8))

    def attend(query):
        return scaled_dot_product_attention(query, query, query, causal=True)

    run = program(attend)
    ran: dict[str, list[jax.Array]] = {"fast": [], "accurate": []}
    compiled: dict[str, list[int]] = {"fast": [], "accurate": []}
    for form in ("fast", "accurate", "fast", "accurate"):
        with generation.run_precision(Numerics(forms=form)), Compiles("attend") as compiles:
            ran[form].append(run(query))
        compiled[form].append(sum(found.kind == "compiled" for found in compiles.programs))
    assert all(first >= 1 and again == 0 for first, again in compiled.values()), compiled
    for form, outputs in ran.items():
        with generation.run_precision(Numerics(forms=form)):
            expected = jax.jit(attend)(query)
        for output in outputs:
            np.testing.assert_array_equal(output, expected)


def test_a_released_program_drops_its_executables_and_compiles_again():
    """`options.release` drops every live program's executables, which jit's cache clearing never reaches
    (XLA:CPU maps each one's code into the process, and a program keeps them for as long as it lives): the
    executable a call compiled is freed once nothing else holds it, and the next call takes a new one (a
    compile, or a load from the persistent compilation cache) and computes what the first did."""
    from dew.nn.kernels import options

    def doubled_sine(x):
        return jnp.sin(x) * 2.0

    run = program(doubled_sine)
    x = jnp.arange(8.0)
    first = run(x)
    held = weakref.ref(run.executable(x))
    options.release()
    gc.collect()
    assert held() is None
    np.testing.assert_array_equal(run(x), first)


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


def test_calls_compiled_together_run_their_executables_after_and_compile_nothing_more():
    """Calls of programs within `compiled_together` answer with their results' shapes and run nothing,
    each traced on this thread; as the block ends each is compiled, all at once, and kept by its program,
    so the same calls after it compile nothing and answer as a call compiled alone does; a call whose
    executable the program keeps lowers nothing again within a block, a call the block lowered already
    compiles once, and a block within another is the outer one's, waiting on nothing."""
    from dew.nn.kernels.options import compiled_together

    run = program(lambda x: x * 2 + 1)
    other = program(lambda x: jnp.sin(x))
    nested = program(lambda x: x - 1)
    two, three, four, five = jnp.ones(2), jnp.ones(3), jnp.ones(4), jnp.ones(5)

    def compiled(programs) -> int:
        return [program.kind for program in programs].count("compiled")

    with Compiles(None) as compiles, compiled_together():
        shapes = [run(two), run(three), other(four), run(two)]
    assert [(shape.shape, shape.dtype) for shape in shapes] == [((2,), jnp.float32), ((3,), jnp.float32),
                                                                ((4,), jnp.float32), ((2,), jnp.float32)]
    assert compiled(compiles.programs) == 3
    with Compiles(None) as again:
        outputs = [run(two), run(three), other(four)]
        with compiled_together():
            run(two)
    assert [program for program in again.programs if program.kind in ("compiled", "lowered")] == []
    np.testing.assert_array_equal(outputs[1], jnp.full(3, 3.0))
    np.testing.assert_array_equal(outputs[2], jnp.sin(jnp.ones(4)))
    with Compiles(None) as joined, compiled_together():
        nested(five)
        with compiled_together():
            nested(five)
    assert compiled(joined.programs) == 1


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class Slots:
    """A state of two arrays, as a server's resident slots are, placed by a tree of its own shape."""

    rows: jax.Array
    columns: jax.Array


def test_a_program_donates_its_state_and_places_it_by_a_tree_of_formats():
    """A program a server steps its state with (`program`, row 3327) donates the state it is handed
    (`donate_argnums`, `donate_argnames`), the state's arrays deleted once it ran, and places its arguments
    and results by a tree of formats of the state's own structure, each array in the layout its format
    names."""
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    formats = Slots(Format(Layout((0, 1)), sharding), Format(Layout((1, 0)), sharding))

    def stepped(slots: Slots, scale):
        return Slots(slots.rows + scale, slots.columns * scale)

    run = program(stepped, donate_argnums=0, in_shardings=(formats, None), out_shardings=formats)
    slots = Slots(jax.device_put(jnp.ones((2, 3)), formats.rows),
                  jax.device_put(jnp.ones((2, 3)), formats.columns))
    stepped_slots = run(slots, 2.0)
    assert slots.rows.is_deleted() and slots.columns.is_deleted()
    assert [leaf.format.layout.major_to_minor for leaf in (stepped_slots.rows, stepped_slots.columns)] == [
        (0, 1), (1, 0)]
    np.testing.assert_array_equal(np.asarray(stepped_slots.rows), np.full((2, 3), 3.0))
    named = program(lambda scale, slots: stepped(slots, scale), donate_argnames="slots",
                    out_shardings=formats)
    again = named(2.0, slots=stepped_slots)
    assert stepped_slots.rows.is_deleted() and stepped_slots.columns.is_deleted()
    np.testing.assert_array_equal(np.asarray(again.columns), np.full((2, 3), 4.0))


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


def _field(number: int, value: bytes | int) -> bytes:
    """One protobuf field: a varint, or bytes length-delimited."""
    def varint(n: int) -> bytes:
        out = b""
        while True:
            out += bytes([(n & 0x7F) | (0x80 if n > 0x7F else 0)])
            n >>= 7
            if not n:
                return out

    if isinstance(value, int):
        return varint(number << 3) + varint(value)
    return varint(number << 3 | 2) + varint(len(value)) + value


def _cubin(registers: int, frame: int) -> bytes:
    """An ELF holding one `.nv.info` section stating a kernel's registers and stack frame, as a cubin does."""
    import struct

    names = b"\0.shstrtab\0.nv.info\0"
    info = b"".join(struct.pack("<BBHII", 0x04, attribute, 8, 0, value)
                    for attribute, value in ((0x2F, registers), (0x11, frame)))
    sections_at = 64 + len(names) + len(info)
    header = (b"\x7fELF\x02\x01\x01" + b"\0" * 9
              + struct.pack("<HHIQQQIHHHHHH", 2, 190, 1, 0, 0, sections_at, 0, 64, 0, 0, 64, 3, 1))
    sections = [struct.pack("<IIQQQQIIQQ", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                struct.pack("<IIQQQQIIQQ", 1, 3, 0, 0, 64, len(names), 0, 0, 1, 0),
                struct.pack("<IIQQQQIIQQ", 11, 0x70000000, 0, 0, 64 + len(names), len(info), 0, 0, 4, 0)]
    return header + names + info + b"".join(sections)


def _kernel_proto(name: str, registers: int, frame: int, notes: tuple[tuple[str, str], ...] = (),
                  spilled: int | None = None) -> bytes:
    """A `MosaicGpuKernelProto`, built through the plugin's own schema (`jax.extend.mosaic_gpu`'s
    `kernel_record_class`): its binary, kernel name, launch's shared memory, what ptxas noted of a loss of
    performance (each a code and a message) and the bytes it spilled, where measured."""
    from jax.extend import mosaic_gpu

    record = mosaic_gpu.kernel_record_class()(version=4, gpu_binary=_cubin(registers, frame),
                                              kernel_name=f"{name}_mosaic_gpu_kernel", smem_bytes=1024)
    for code, message in notes:
        record.assembler_notes.add(code=code, message=message)
    if spilled is not None:
        record.spilled_bytes = spilled
    return record.SerializeToString()


def _thunk(calls: int, kernels: list[bytes]) -> bytes:
    """A thunk's proto naming `calls` Mosaic GPU custom calls, each kernel proto in a message of its own."""
    return b"".join(_field(2, _field(1, b"mosaic_gpu_v2")) for _ in range(calls)) + b"".join(
        _field(3, _field(5, kernel)) for kernel in kernels)


def test_a_thunk_whose_kernel_binary_is_missing_refuses_rather_than_reading_as_no_spill():
    """Each Mosaic GPU call a thunk names has its binary (`assembled.thunk_kernels`): two calls and two kernel
    protos read as two kernels, their registers and frames their cubins'; two calls and one is refused, as a
    missing binary would read as a kernel that does not spill (row 2851)."""

    both = assembled.thunk_kernels(_thunk(2, [_kernel_proto("dew_gmm__l0123456789", 168, 0),
                                           _kernel_proto("dew_tgmm", 255, 344)]))
    read = [(kernel.kernel, kernel.launch, kernel.registers, kernel.frame, kernel.shared) for kernel in both]
    assert read == [("dew_gmm", "0123456789", 168, 0, 1024), ("dew_tgmm", None, 255, 344, 1024)]
    with pytest.raises(ValueError):
        assembled.thunk_kernels(_thunk(2, [_kernel_proto("dew_gmm", 168, 0)]))


def test_a_kernel_is_refused_for_what_ptxas_spilled_not_for_its_stack_frame():
    """A kernel whose stack frame holds its own locals and nothing spilled (a prologue's sine: libdevice's
    range reduction keeps a 32-byte array, row 9649), as ptxas measured it (its record's `spilled_bytes`), is
    certified; one that spilled is refused; one a build measured nothing of is held to its frame, which
    holds every byte it spills."""
    kernels = assembled.thunk_kernels(_thunk(3, [_kernel_proto("dew_gmm", 168, 32, spilled=0),
                                              _kernel_proto("dew_tgmm", 255, 96, spilled=64),
                                              _kernel_proto("dew_added", 168, 32)]))
    assert [(kernel.spills, kernel.refused) for kernel in kernels] == [(0, False), (64, True), (32, True)]


def test_a_kernels_assembler_notes_are_read_from_its_compiled_call():
    """What ptxas noted of a loss of performance as it assembled a kernel (Dew's JAX build keeps each note's
    code and message in its compiled call, its record's `assembler_notes`) is read with its binary, in order:
    an sm90 attention backward's warpgroup products serialized for want of registers, where nothing spills,
    is data a certification refuses by, not text a process printed. A kernel noted nothing reads none."""
    serialized = ("C7512", "wgmma.mma_async instructions are serialized due to insufficient register "
                  "resources for the function 'attention_dkv'")
    ignored = ("C7507", "'setmaxnreg' ignored to maintain minimum register requirements")
    kernels = [_kernel_proto("attention_dkv", 168, 0, (serialized, ignored)),
               _kernel_proto("attention_dq", 128, 0)]
    noted, plain = assembled.thunk_kernels(_thunk(2, kernels))
    assert noted.notes == (assembled.AssemblerNote(*serialized), assembled.AssemblerNote(*ignored))
    assert plain.notes == ()


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize("statics", [False, True], ids=["dynamic", "static"])
def test_a_program_called_under_another_mesh_of_its_shape_runs_on_that_meshs_devices(statics):
    """A program called under two meshes of one shape over other devices runs on each mesh's own devices,
    as jit does: its executable is keyed by the mesh in place, its dynamic and its static calls alike
    (ledger 3452), where under the second mesh it ran the first's executable on the first's devices."""
    from jax.sharding import AxisType, Mesh, PartitionSpec

    def function(x, scale=1):
        return jax.lax.with_sharding_constraint(x, PartitionSpec("x")) * scale + 1

    devices = jax.devices()
    meshes = [Mesh(np.asarray(devices[start:start + 2]), ("x",), axis_types=(AxisType.Auto,))
              for start in (0, 2)]
    compiled = program(function, **({"static_argnames": "scale"} if statics else {}))
    given, scale = np.arange(8, dtype=np.float32), 2 if statics else 1
    for mesh in meshes:
        with jax.set_mesh(mesh):
            ran = compiled(given, **({"scale": scale} if statics else {}))
        assert ran.sharding.device_set == set(mesh.devices.flat)
        np.testing.assert_array_equal(np.asarray(ran), given * scale + 1)


def test_a_programs_spilling_triton_launch_is_recorded_with_no_mosaic_kernel_beside_it():
    """A program of one Pallas-Triton launch, no Mosaic GPU kernel in it, compiled for a compile-only H100:
    its launch's spill (a row softmax of a 64 x 512 block at one warp, 1024 values a thread) is read from its
    binary and recorded as any template launch's is (`report.misses`), whatever kernels share its program
    (row 3960: the compile read binaries only beside a Mosaic kernel)."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as pltriton
    from jax.sharding import NamedSharding, PartitionSpec

    from dew.nn.kernels.generation import compile_only_gpus

    def softmax(x_ref, o_ref):
        x = x_ref[...]
        exponentials = jnp.exp(x - jnp.max(x, axis=1, keepdims=True))
        o_ref[...] = exponentials / jnp.sum(exponentials, axis=1, keepdims=True)

    shape = jax.ShapeDtypeStruct((64, 512), jnp.float32)

    def program(x):
        return pl.pallas_call(softmax, out_shape=shape, grid=(1,), name="dew_softmax__l0123456789",
                              compiler_params=pltriton.CompilerParams(num_warps=1, num_stages=1))(x)

    devices = jax.sharding.Mesh(np.array(compile_only_gpus("sm90")), ("data",))
    spec = jax.ShapeDtypeStruct((64, 512), jnp.float32, sharding=NamedSharding(devices, PartitionSpec()))
    def lowered():
        return jax.jit(program).trace(spec).lower()

    with (report.refusals(), generation.assumed("sm90"),
          jax.sharding.use_abstract_mesh(generation.gpu_mesh(devices, "sm90"))):
        report.compiled(lowered(), relower=lowered)
        missed = [(miss.kernel, miss.spilled > 0) for miss in report.misses()]
    assert missed == [("dew_softmax", True)]


def _paired_step(launch):
    """A compile-only H100's mesh, and a lowering of one paired SwiGLU step through `launch` over it, traced
    afresh at each call (a compile's `relower`)."""
    from jax.sharding import NamedSharding, PartitionSpec

    from dew.nn.kernels.generation import compile_only_gpus

    devices = jax.sharding.Mesh(np.array(compile_only_gpus("sm90")), ("data",))
    operands = [jax.ShapeDtypeStruct(shape, jnp.bfloat16, sharding=NamedSharding(devices, PartitionSpec()))
                for shape in ((ROWS, FEATURES), (EXPERTS, FEATURES, WIDTH), (EXPERTS, FEATURES, WIDTH))]

    def lowered():
        def stepped(x, gate, up):
            sizes = jnp.full((EXPERTS,), ROWS // EXPERTS, jnp.int32)
            return launch(x, gate, sizes, paired=up, out_dtype="bfloat16", trans_rhs=False,
                          hooks=Hooks(epilogue=lambda g, u: jax.nn.silu(g) * u),
                          interpret=False)
        return jax.jit(stepped).trace(*operands).lower()

    return devices, lowered


class TwiceUncounted(Uncounted):
    """`Uncounted`, its widest tiles its only ones until a compile has refused two of its configs: a launch
    whose model misses at two configs in turn."""

    def candidate(self, ba, config, target):
        return grouped_matmul.MosaicGmm.candidate(self, ba, config, target) and (
            config.block_n == 256 or len(report.refused_launches()) >= 2)


@lowers("sm90")
def test_a_launch_spilling_at_two_configs_steps_down_alone_and_its_program_compiles_whole_once_more():
    """A program of one launch whose kernel spills at its first two configs, compiled for a compile-only H100
    (`report.compiled`) with its launches certified there (`report.certifying`, as a process on an H100
    certifies them on its own): each config is refused as the launch compiles alone before its program
    traces (`report.certified`), and the program compiles whole once, at the config that holds, where a
    step-down after its compile compiled it whole each step (row 5359: about 18 s of a step-down's 30 on an
    sm100 step); the program kept spills nothing, as the launch's last step alone did not: a kernel's code
    alone is its code in the program, the fact certification rests on."""
    from dew.nn.kernels.generation import compile_only_gpus

    devices, lowered = _paired_step(TwiceUncounted())
    with (report.refusals(), generation.assumed("sm90"), Compiles("stepped") as compiles,
          report.certifying(compile_only_gpus("sm90")[0]),
          jax.sharding.use_abstract_mesh(generation.gpu_mesh(devices, "sm90"))):
        report.compiled(lowered(), relower=lowered)
        misses = report.misses()
    assert [miss.stepped for miss in misses] == [True, True], misses
    assert len({miss.config for miss in misses}) == 2 and all(miss.spilled for miss in misses), misses
    assert compiles.count == 1


@lowers("sm90")
def test_a_program_lowered_before_another_compile_refused_its_config_is_relowered_past_it():
    """Two lowerings of one step through a launch whose first tile spills (`Uncounted`), both traced before
    either compiles, compiled for a compile-only H100 one after the other (`report.compiled`), as another
    thread's compile or a certification elsewhere commits a refusal after a program's trace took the config:
    the second, whose compile spills at the config the first's refused, is lowered afresh past it, so no
    program kept spills (no miss left unstepped), where its spill read as already held kept it (row 8603)."""
    devices, lowered = _paired_step(Uncounted())
    with (report.refusals(), generation.assumed("sm90"),
          jax.sharding.use_abstract_mesh(generation.gpu_mesh(devices, "sm90"))):
        first, second = lowered(), lowered()
        report.compiled(first, relower=lowered)
        report.compiled(second, relower=lowered)
        misses = report.misses()
    assert misses and all(miss.stepped for miss in misses), misses


def _aside(devices, lowered_first, relowered):
    """`report.lowered_aside` of a lowering's first call `lowered_first` and its retry `relowered` over
    `devices`, its launches certified aside on a compile-only H100 within a `refusals` scope of its own: its
    misses, its refused launches and the retries it made."""
    from dew.nn.kernels.generation import compile_only_gpus

    retries: list[int] = []

    def retried():
        retries.append(1)
        return relowered()

    with (report.refusals(), generation.assumed("sm90"), report.certifying(compile_only_gpus("sm90")[0]),
          jax.sharding.use_abstract_mesh(generation.gpu_mesh(devices, "sm90"))):
        report.lowered_aside(lowered_first, retried, lambda value: value)
        return report.misses(), report.refused_launches(), retries


@lowers("sm90")
def test_an_aside_miss_is_recorded_stepped_in_the_recording_that_queued_its_launch():
    """A program whose launch spills at its first tile (`Uncounted`), lowered with its launches certified
    aside (`report.lowered_aside`), its first lowering within a recording of its own: the launch's miss is
    recorded once, stepped by its retry's certification past the tile, in the recording its launch was queued
    in though the block settles outside it (row 11786's review: misses went to the draining context's)."""
    devices, lowered = _paired_step(Uncounted())
    queued: list[report.Recorded] = []

    def first():
        with report.recording() as recorded:
            queued.append(recorded)
            return lowered()

    misses, _, retries = _aside(devices, first, lowered)
    assert misses and all(miss.stepped for miss in misses) and retries, (misses, retries)
    assert tuple(queued[0].missed) == misses, (queued[0].missed, misses)


@lowers("sm90")
def test_an_aside_refusal_stays_in_the_refusals_scope_that_queued_its_launch():
    """The same program's first lowering within a `refusals` scope of its own: the launch it queued refuses
    its tile in that scope as the block settles outside it, so the enclosing scope holds no refusal and no
    miss and retries nothing (row 11786's review: the refusal went to the draining context's registry, which
    retried the lowering past it)."""
    devices, lowered = _paired_step(Uncounted())

    def first():
        with report.refusals():
            return lowered()

    misses, refused, retries = _aside(devices, first, lowered)
    assert not misses and not refused and not retries, (misses, refused, retries)


class SpillingOnly(Uncounted):
    """`Uncounted`'s first tile alone, which spills in every compile, and no tile once a compile refused it: a
    launch whose every legal config spills."""

    def candidate(self, ba, config, target):
        return Uncounted.candidate(self, ba, config, target) and not report.refused_launches()


@lowers("sm90")
def test_a_program_whose_launch_spills_at_every_config_left_is_refused_not_kept():
    """A program whose launch spills at every config its derivation has, compiled for a compile-only H100
    (`report.compiled`) with its launches certified there (`report.certifying`, as a process on an H100
    certifies them on its own): the compile raises (`report.Exhausted`) and keeps no executable, as a derived
    kernel never spills (docs/design/kernel-templates.md), its misses recorded as not stepped, its launch
    having no config to step to (audit K2-6: the program found spilling was kept; row 6162: certification
    recorded them stepped)."""
    from dew.nn.kernels.generation import compile_only_gpus

    devices, lowered = _paired_step(SpillingOnly())
    with (report.refusals(), generation.assumed("sm90"), report.certifying(compile_only_gpus("sm90")[0]),
          jax.sharding.use_abstract_mesh(generation.gpu_mesh(devices, "sm90"))):
        with pytest.raises(report.Exhausted):
            report.compiled(lowered(), relower=lowered)
        misses = report.misses()
    assert misses and not any(miss.stepped for miss in misses), misses


class TwoKernels(grouped_matmul.MosaicGmm):
    """The grouped forward launching its kernel again over half its rows: one launch of two Mosaic GPU
    kernels of the one name its launch gives them (`report.named`)."""

    @functools.wraps(grouped_matmul.MosaicGmm._fwd)  # its binding reads its forward's signature
    def _fwd(self, tokens, kernel, sizes, *args, **kwargs):
        whole, residuals = super()._fwd(tokens, kernel, sizes, *args, **kwargs)
        half, _ = super()._fwd(tokens[: tokens.shape[0] // 2], kernel, sizes // 2, *args, **kwargs)
        return whole.at[: half.shape[0]].add(half), residuals


@lowers("sm90")
def test_two_kernels_of_one_launch_are_two_names_each_certified_on_its_own():
    """A launch of two kernels its template names alike (the grouped forward and its re-launch over half its
    rows, both `dew_gmm`, as a training call's keys-alone re-forming beside its weighted launch were both
    `recurrence_weighted`) is two names, each kernel's form apart (`report.forming`: the half call's shape),
    so a start certifying it on a compile-only H100 reads each kernel's evidence on its own
    (`report.certified`, whose read of one, the names alike, missed the other's 8-byte spill) and a step-down
    compiles each alone (`report.compiled_alone`)."""
    from dew.nn.kernels.generation import compile_only_gpus
    from dew.nn.kernels.lowered import kernel_calls

    op = TwoKernels()
    devices, lowered = _paired_step(op)
    with generation.assumed("sm90"), jax.sharding.use_abstract_mesh(generation.gpu_mesh(devices, "sm90")):
        with report.certifying(compile_only_gpus("sm90")[0]):
            program = lowered()
        names = [report.launch_name(call.name, call.launch) for call in kernel_calls(program, unrun=True)
                 if call.kind == "mosaic_gpu"]
        alone = report.compiled_alone(program, None, lambda call: True)
    # Certification lowers the launch in a program of its own, whose launch it names by its own placement.
    [certified] = [each for each in report.certifications() if each.op == type(op).__name__]
    assert len(set(names)) == len(names) == len(certified.kernels) == 2, (names, certified.kernels)
    assert set(alone) == set(names), alone


@lowers("sm90")
def test_a_launch_traced_under_a_compile_only_gpus_mesh_is_certified_on_that_gpu():
    """A launch traced under a compile-only H100's mesh (`jax.set_mesh`, as a trainer traces its step's
    programs), no device named to certify on, is certified on that mesh's GPU (`report.certified`), the one a
    start's compile ahead certifies it on: one certification of the launch, wherever the process runs. Where
    a host's own GPU of the generation certified it instead, a start compiling ahead on that host certified
    each launch traced outside its certifying block twice, apart (row 6791: the GH200's recurrent step
    certified 5 of its 28 programs twice), and a host of no GPU certified none of them."""
    class Meshed(TwoKernels):
        """`TwoKernels`, a launch of its own: certified here, whatever this process certified before (row
        10853: run after the file's other tests, `TwoKernels` was certified already and this saw none)."""

    op = Meshed()
    devices, lowered = _paired_step(op)
    before = {id(each) for each in report.certifications()}
    with generation.assumed("sm90"), jax.set_mesh(devices):
        lowered()
    [certified] = [each for each in report.certifications()
                   if id(each) not in before and each.op == type(op).__name__]
    assert certified.kernels, certified


@lowers("sm90")
def test_a_runs_launch_traced_outside_its_mesh_is_certified_on_the_gpu_it_targets():
    """A launch a run traces outside its mesh (its start's, `Trainer._started`) is certified on the GPU the
    run targets (`report.targeting`, a compile-only H100's mesh) and on no other, on any host. Certified on
    a host's own GPU, a GH200's start compiling ahead for an H100 certified five launches twice, apart, its
    step's trace again on the mesh's (row 6824)."""
    class Targeted(TwoKernels):
        """`TwoKernels`, a launch of its own: certified here, whatever this process certified before."""

    op = Targeted()
    devices, lowered = _paired_step(op)
    before = {id(each) for each in report.certifications()}
    with generation.assumed("sm90"), report.targeting(devices):
        lowered()
    [certified] = [each for each in report.certifications()
                   if id(each) not in before and each.op == type(op).__name__]
    assert certified.device == devices.devices.flat[0], certified.device


@pytest.mark.gpu(reason="certifies and runs a kernel on this host's GPU")
def test_a_plain_jit_on_this_gpu_certifies_its_launch_there_and_runs():
    """A Dew kernel called under a plain jit on this GPU, no mesh and no run naming a target, is certified
    on the device the call runs on (JAX's default device) and runs there: a target a run names
    (`report.targeting`) leaves a plain call's certification where it was. Its arithmetic is
    test_norm_swiglu's."""
    from dew.nn.norms import stream_rms_normalized

    # Rows of no other test's, so this process certifies the launch here.
    x = jax.random.normal(jax.random.key(0), (232, 384), jnp.bfloat16)
    scale = jnp.ones((384,), jnp.bfloat16)
    before = {id(each) for each in report.certifications()}
    jax.jit(lambda x, scale: stream_rms_normalized(
        x, None, scale, 1e-6, jnp.bfloat16, scale_offset=False, scale_after_cast=False, fp32_statistics=True,
        implementation="pallas")[1])(x, scale).block_until_ready()
    certified = [each for each in report.certifications() if id(each) not in before]
    assert certified and {each.device for each in certified} == {jax.devices()[0]}, certified


@lowers("sm90")
def test_a_launchs_rungs_traced_under_a_compile_only_gpus_mesh_are_certified_on_that_gpu():
    """A launch's configs certified in turn (`report.certified`, a ladder's rungs past its first) under a
    compile-only H100's mesh, no device named to certify on: each certified on that mesh's GPU
    (`Certification.device`), which the certifying thread reads, and none on another: never the host's own
    GPU, nor nothing on a host of none, a config no assembler checked kept (train 909's review F4 of row
    6791's fix)."""
    from dew.nn.kernels.generation import compile_only_gpus

    op = TwoKernels()
    devices = jax.sharding.Mesh(np.array(compile_only_gpus("sm90")), ("data",))
    operands = [jax.ShapeDtypeStruct(shape, dtype) for shape, dtype in (
        ((ROWS, FEATURES), jnp.bfloat16), ((EXPERTS, FEATURES, WIDTH), jnp.bfloat16),
        ((EXPERTS,), jnp.int32))]
    before = {id(each) for each in report.certifications()}
    with generation.assumed("sm90"), jax.set_mesh(devices):
        bound = op.bind(*operands, out_dtype="bfloat16", trans_rhs=False, interpret=False)
        first = bound.heuristics_config
        configs = [first, dataclasses.replace(first, stages=first.stages - 1)]
        answers = [report.certified(op, bound, config) for config in configs]
    assert all(answer is not None for answer in answers), answers
    found = [each for each in report.certifications() if id(each) not in before]
    assert sorted(map(repr, (each.config for each in found))) == sorted(map(repr, configs)), found
    assert {each.device for each in found} == set(devices.devices.flat) and all(
        each.kernels for each in found), found


def test_two_kernels_of_one_name_and_launch_are_refused_not_read_as_one():
    """Two Mosaic GPU calls of one name and launch whose kernels differ (a kernel whose body reads what its
    name holds not, neither its launch's config nor its form) are refused wherever a kernel's evidence is
    read by its name (`report.kernel_bodies`), as each record would hold one kernel's evidence for both; one
    kernel called twice is one."""
    from dew.nn.kernels.lowered import KernelCall

    def call(kernel_hash: str) -> KernelCall:
        return KernelCall("mosaic_gpu", "dew_gmm", b"", kernel_hash, None, launch="0123456789")

    assert set(report.kernel_bodies([call("a"), call("a")])) == {"dew_gmm__l0123456789"}
    with pytest.raises(report.KernelsOfOneName):
        report.kernel_bodies([call("a"), call("b")])


@lowers("sm90")
def test_one_kernel_called_from_two_places_compiles_alone_once():
    """A Mosaic GPU kernel and a Pallas-Triton kernel each called from two places under two name scopes, its
    calls located apart, are each one kernel of its name: a step-down compiles each alone once
    (`report.compiled_alone`), refusing neither."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as pltriton
    from jax.sharding import NamedSharding, PartitionSpec
    from kernel_compile import staged_copy

    from dew.nn.kernels.generation import compile_only_gpus

    def added(x_ref, y_ref, o_ref):
        o_ref[...] = x_ref[...] + y_ref[...]

    shape = jax.ShapeDtypeStruct((128, 128), jnp.float32)
    copy = staged_copy((128, 128), jnp.float32)

    def summed(x, y):
        return pl.pallas_call(added, out_shape=shape, grid=(1,), name="dew_added__l0123456789",
                              compiler_params=pltriton.CompilerParams(num_warps=4, num_stages=1))(x, y)

    def program(x, y):
        with jax.named_scope("first"):
            first = summed(x, y) + copy(x)
        with jax.named_scope("second"):
            return first + summed(y, x) + copy(y)

    devices = jax.sharding.Mesh(np.array(compile_only_gpus("sm90")), ("data",))
    spec = jax.ShapeDtypeStruct((128, 128), jnp.float32, sharding=NamedSharding(devices, PartitionSpec()))
    with generation.assumed("sm90"), jax.sharding.use_abstract_mesh(generation.gpu_mesh(devices, "sm90")):
        lowered = jax.jit(program).trace(spec, spec).lower()
        alone = report.compiled_alone(lowered, None, lambda call: True)
    assert alone is not None and len(alone) == 2 and all(found.error is None for found in alone.values())


@lowers("sm90")
@pytest.mark.parametrize("constant", [False, True], ids=["no-constant", "a-constant"])
def test_an_executable_carries_each_mosaic_and_triton_kernels_binary_once(constant):
    """A program of a Pallas-Triton launch feeding a Mosaic GPU kernel and an XLA fusion, compiled for an
    H100, with or without a constant: `assembled.binaries` reads the launch once, its cubin from its custom
    kernel's proto (a program holding a constant gives every thunk that constant's module alone as its
    binary, which states no registers: row 3858), and the Mosaic kernel once, from its kernel proto, each
    stating its registers; the fusion, whose thunk carries its cubin too, names no launch and is no
    template's."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as pltriton
    from jax.sharding import NamedSharding, PartitionSpec
    from kernel_compile import staged_copy

    from dew.nn.kernels import generation
    from dew.nn.kernels.generation import compile_only_gpus

    def added(x_ref, y_ref, o_ref):
        o_ref[...] = x_ref[...] + y_ref[...]

    copy = staged_copy((128, 128), jnp.float32)
    shape = jax.ShapeDtypeStruct((128, 128), jnp.float32)
    table = np.arange(128 * 128, dtype=np.float32).reshape(128, 128) if constant else 0

    def program(x, y):
        launched = pl.pallas_call(added, out_shape=shape, grid=(1,), name="dew_added__l0123456789",
                                  compiler_params=pltriton.CompilerParams(num_warps=4, num_stages=1))(x, y)
        return jnp.concatenate([copy(launched), jnp.sin(x) + table]).sum(axis=1)

    devices = jax.sharding.Mesh(np.array(compile_only_gpus("sm90")), ("data",))
    spec = jax.ShapeDtypeStruct((128, 128), jnp.float32, sharding=NamedSharding(devices, PartitionSpec()))
    with generation.assumed("sm90"), jax.sharding.use_abstract_mesh(generation.gpu_mesh(devices, "sm90")):
        compiled = jax.jit(program).trace(spec, spec).lower().compile()
    found = assembled.binaries(compiled)
    assert len(found) == 2 and all(binary.registers > 0 and not binary.spills for binary in found)
    launched = [(binary.kernel, binary.launch) for binary in found if binary.launch]
    assert launched == [("dew_added", "0123456789")]

def test_a_program_its_caller_owns_is_released_with_its_caller():
    """A program over a closure or a bound method its caller owns (`owned_program`), as a run's placement
    and its host phases compile, keeps no reference of Dew's: once the caller lets it go, the owner the
    closure holds (a trainer, its objective and their arrays) is released, as no process-wide cache holds
    the closure. A program over a module's function (`program`) is kept for every caller alike."""

    class Owner:
        """A run's stand-in, holding an array its closure reads."""

        def __init__(self):
            self.held = jnp.arange(4.0)

        def scaled(self, x):
            return x * self.held

    def reading(held: Owner):
        return lambda x: x + held.held

    owner = Owner()
    released = weakref.ref(owner)
    closure = owned_program(reading(owner))
    bound = owned_program(owner.scaled)
    np.testing.assert_array_equal(np.asarray(closure(jnp.ones(4))), np.arange(4.0) + 1)
    np.testing.assert_array_equal(np.asarray(bound(jnp.ones(4))), np.arange(4.0))
    del owner, closure, bound
    gc.collect()
    assert released() is None
    assert program(jnp.sum) is program(jnp.sum)


def test_a_fetched_program_runs_the_executable_of_its_first_call():
    """A program a host loop calls every step (`Fetched`) answers as the program does for arguments of its
    first call's shapes and dtypes, however their values change; arguments of another shape are refused
    where the program itself would compile for them; and a program with a static argument, whose
    executable takes the dynamic arguments alone, is refused at once."""
    doubled = owned_program(lambda x: x * 2.0)
    fetched = Fetched(doubled)
    for value in (1.0, 3.0):
        np.testing.assert_array_equal(np.asarray(fetched(jnp.full(3, value))), np.full(3, 2 * value))
    with pytest.raises(TypeError):
        fetched(jnp.ones(4))
    np.testing.assert_array_equal(np.asarray(doubled(jnp.ones(4))), np.full(4, 2.0))
    with pytest.raises(ValueError):
        Fetched(owned_program(lambda scale, x: scale * x, static_argnums=0))


def test_a_fetched_step_called_in_another_trace_context_runs_that_contexts_program():
    """A fetched step (`Fetched`) keeps an executable for each trace context it is called in, as the program
    compiles one for each: a step first run in float32 and then with 64-bit types on runs the program that
    context traces (its default float the wider one), not the executable it fetched first."""
    # `float`, the default float: float32 with x64 off, float64 with it on (no float64 asked for).
    widened = Fetched(owned_program(lambda x: jnp.asarray(x, float) * 3.0))
    narrow = jnp.ones(3, jnp.float32)
    assert widened(narrow).dtype == jnp.float32
    with jax.enable_x64(new_val=True):
        assert widened(narrow).dtype == jnp.float64


def test_a_program_within_another_takes_the_enclosing_programs_options():
    """A program called within another trace (a kernel's own program inside a step) is the enclosing
    program's part and compiles under its options, as JAX takes compiler options on the outermost jit
    alone: a program opened with options is called inside another without refusal."""
    with compiled_with({"xla_cpu_enable_fast_math": False}):
        inner = owned_program(lambda x: x * 2.0)
        outer = owned_program(lambda x: inner(x) + 1.0)
        np.testing.assert_array_equal(np.asarray(outer(jnp.ones(3))), np.full(3, 3.0))

def test_a_program_under_an_eager_transformation_compiles_under_its_own_options():
    """A program called under an eager `jax.vmap` or `jax.grad`, which no enclosing program compiles,
    compiles under its own options as a direct call does, so an option XLA does not know is refused
    there too; within another program's trace it is that program's part and takes its options."""
    with compiled_with({"xla_dew_no_such_option": True}):
        unknown = owned_program(lambda x: x * 2.0)
    rows = jnp.ones((2, 3))
    for called in (lambda: unknown(rows), lambda: jax.vmap(unknown)(rows),
                   lambda: jax.grad(lambda x: jnp.sum(unknown(x)))(rows)):
        with pytest.raises(jax.errors.JaxRuntimeError):
            called()
    enclosing = owned_program(lambda x: unknown(x) + 1.0)
    np.testing.assert_array_equal(np.asarray(enclosing(rows)), np.full((2, 3), 3.0))


CONVERTING_LISTING = """
//--------------------- .text.dew_gmm_mosaic_gpu_kernel --------------------------
.text.dew_gmm_mosaic_gpu_kernel:
        /*0000*/                   S2R R10, SR_TID.X ;
        /*0010*/                   MOV R9, 0x400 ;
.L_x_1:
        /*0020*/                   R2UR UR4, R9 ;
        /*0030*/                   IADD3 R2, R2, 0x1, RZ ;
        /*0040*/                   HGMMA.64x128x16.F32.BF16 R24, gdesc[UR4], RZ ;
        /*0050*/                   ISETP.NE.AND P0, PT, R2, 0x8, PT ;
        /*0060*/               @P0 BRA `(.L_x_1) ;
        /*0070*/                   EXIT ;
"""
"""An MMA loop converting a register to a uniform one each iteration, no waterfall around the conversion."""

WATERFALL_LISTING = """
.text.dew_gmm_mosaic_gpu_kernel:
        /*0000*/                   S2R R10, SR_TID.X ;
.L_x_2:
        /*0010*/                   ELECT P1, UR6, PT ;
        /*0020*/                   R2UR UR8, R10 ;
        /*0030*/                   UBLKCP.S.G [UR8], [UR4], UR5 ;
        /*0040*/                   PLOP3.LUT P1, PT, P1, PT, PT, 0x80, 0x8 ;
        /*0050*/               @P1 BRA.U.ANY `(.L_x_2) ;
        /*0060*/                   EXIT ;
"""
"""A copy whose operand ptxas could not prove uniform, issued lane by lane in an elect waterfall (BRA.U.ANY)
around its R2UR (notes/ptxas-re.md §2: a divergent copy operand)."""

DEAD_MMA_LISTING = """
.text.negated_mosaic_gpu_kernel:
        /*0570*/                   WARPGROUP.ARRIVE ;
        /*0580*/                   HGMMA.64x8x16.F16 RZ, gdesc[URZ], RZ, !UPT, gsb0 ;
        /*0590*/                   WARPGROUP.DEPBAR.LE gsb0, 0x0 ;
        /*05b0*/                   STS.64 [R3+0x4000], R16 ;
        /*05c0*/                   EXIT ;
"""
"""ptxas 12.9's listing of a negated warpgroup product (row 4346's minimal kernel at base): its four MMAs
dropped for one writing no register, the stored tile a constant."""


REGISTER_A_LISTING = """
.text.dew_gmm_mosaic_gpu_kernel:
.L_x_9:
        /*1c20*/                   LEA R131, R20, UR16, 0x2 ;
        /*1c40*/                   LDSM.16.M88.4 R124, [R131] ;
        /*2600*/                   WARPGROUP.ARRIVE ;
        /*2630*/                   HGMMA.64x192x16.F32.BF16 R24, R124, gdesc[UR24].tnspB, R24, gsb0 ;
        /*26a0*/                   WARPGROUP.DEPBAR.LE gsb0, 0x1 ;
        /*27e0*/               @P1 BRA `(.L_x_9) ;
        /*2810*/                   WARPGROUP.DEPBAR.LE gsb0, 0x0 ;
"""
"""Row 5416's prologued forward at 2497d1526ec (dew_gmm__lc7f44bb880, its loop cut to one product): a
register-A warpgroup MMA's group kept pending (`DEPBAR.LE gsb0, 0x1`) across the loop's back edge, where
the next trip's load writes its operand A (R124)."""


DEAD = "warpgroup MMA writing no register"
REUSED = "register operand A before its group retired"


@pytest.mark.parametrize(("listing", "found"), [
    (WATERFALL_LISTING, "waterfall"),
    (DEAD_MMA_LISTING, DEAD),
    (DEAD_MMA_LISTING.replace("HGMMA.64x8x16.F16", "QGMMA.64x8x32.F32.E4M3.E4M3"), DEAD),
    (DEAD_MMA_LISTING.replace("HGMMA.64x8x16.F16", "@P0 IGMMA.64x8x32.S32.S8.S8"), DEAD),
    (REGISTER_A_LISTING, REUSED),
])
def test_the_sass_rules_find_what_ptxas_left_of_a_kernel(listing, found):
    """The compile gate's SASS rules read nvdisasm's listing of a kernel (row 4071, the kernel challenge's
    harness/ptxas checks): an elect waterfall (a backward BRA.U.ANY) holding an R2UR serializes a copy whose
    operand ptxas could not prove uniform, and a warpgroup MMA writing no register (an HGMMA, an fp8 QGMMA or
    an integer IGMMA whose destination is RZ) is what ptxas leaves of products it found never read (row 4346);
    a loop writing a register-A MMA's operand while its group is pending is undefined behaviour (row 5416);
    each is one finding the gate fails, and a loop's conversion with none is none, as is the register-A loop
    that retires its group before the back edge."""
    (finding,) = sass.sass_findings_of(listing)
    assert found in finding
    assert sass.sass_findings_of(CONVERTING_LISTING) == ()
    assert sass.sass_findings_of(REGISTER_A_LISTING.replace("gsb0, 0x1", "gsb0, 0x0")) == ()
    # A retire only some trips take (a predicated wait) retires nothing the walk can count on.
    conditional = REGISTER_A_LISTING.replace("WARPGROUP.DEPBAR.LE gsb0, 0x1",
                                             "@P0 WARPGROUP.DEPBAR.LE gsb0, 0x0")
    (finding,) = sass.sass_findings_of(conditional)
    assert "register operand A before its group retired" in finding


def test_the_compile_gate_fails_a_kernel_above_the_register_cap_its_launch_runs_at():
    """A kernel allocating more registers a thread than the cap its launch runs at (its record's `cap`, as its
    compile states it: its consumers' raise, else its threads' share of the multiprocessor's registers at the
    programs it is launched to hold), spilling nothing, is a kernel the gate fails, and one at its cap passes
    (row 6670): an allocation above the launch's leaves a multiprocessor fewer programs than it was planned
    for. One whose record states no cap fails too, where its allocation went unjudged."""
    import kernel_compile as kc

    held = kc.Report("kernel", 1024, 0, 168, 0, cap=168)
    assert held.failure(232448, 512) is None
    over = dataclasses.replace(held, registers=192)
    assert over.failure(232448, 512) is not None, over
    # A record stating no cap (no thread count, no raise) is no evidence it holds its launch's (row 6670).
    uncapped = dataclasses.replace(held, cap=None)
    assert uncapped.failure(232448, 512) is not None, uncapped


CONSUMERS_SPILL_LISTING = """
.text.recurrence_scores_back_mosaic_gpu_kernel:
        /*0000*/                   LDC R1, c[0x0][0x28] ;
        /*0e60*/               @P2 BRA `(.L_x_0) ;
.L_x_1:
        /*0e80*/                   USETMAXREG.TRY_ALLOC.CTAPOOL UP0, 0xe8 ;
        /*0ea0*/              @!P1 BRA `(.L_x_1) ;
        /*27b0*/                   LDS.64 R192, [R25+0xa4a0] ;
        /*27c0*/                   STL [R1], R173 ;
        /*3950*/                   LDL.LU R203, [R1] ;
        /*9a30*/               @P1 BRA `(.L_x_13) ;
        /*9a50*/                   USETMAXREG.DEALLOC.CTAPOOL 0x18 ;
        /*9a60*/                   S2R R4, SR_CTAID.X ;
"""
"""An H100's write-gated scores' backward at its producer's least (the termwise sums' 47b93b22f; its listing
cut to its roles' setmaxnreg and a spill's store and load): its consumers raised to 232 registers, its
16-byte spill in their code, before the producer's drop to 24."""


def test_a_spills_roles_are_read_from_where_its_setmaxnreg_places_it():
    """A kernel's spill stands in the role whose setmaxnreg precedes it in nvdisasm's listing (row 6744): the
    write-gated scores' backward's after its consumers' raise, its consumers'; the same store and load after
    the producer's drop, as a B200's decayed keys spilled (row 6747), its producer's; both where each role
    spills; neither role's in a prologue before either setmaxnreg."""
    spill = ("        /*27c0*/                   STL [R1], R173 ;\n"
             "        /*3950*/                   LDL.LU R203, [R1] ;\n")
    after_drop = CONSUMERS_SPILL_LISTING.replace(spill, "") + spill
    assert sass.spilled_roles(CONSUMERS_SPILL_LISTING) == (sass.CONSUMERS,)
    assert sass.spilled_roles(after_drop) == (sass.PRODUCER,)
    assert sass.spilled_roles(CONSUMERS_SPILL_LISTING + spill) == (sass.CONSUMERS, sass.PRODUCER)
    assert sass.spilled_roles(spill + CONSUMERS_SPILL_LISTING.replace(spill, "")) == ()


def test_the_compile_gate_reads_machine_code_or_fails_a_kernel_it_could_not_read():
    """The SASS rules' disassembler comes with the compile gate (nvdisasm, the ci extra's, in every lock the
    suite runs on), and a kernel whose machine code went unread (no disassembler) is no kernel the gate
    passes: its report names the machine code unread, which fails it as an unread register count does (row
    4071; train 591)."""
    import kernel_compile as kc

    assert sass.disassembler() is not None
    unread = assembled.Binary("gemm_mosaic_gpu_kernel", None, 168, 0, "mosaic_gpu", 1024, sass=None)
    (found,) = kc.reports({}, [unread])
    assert "machine code" in found.unread
    assert found.failure(232448, 0) is not None


class SpillingFirst(Uncounted):
    """`Uncounted` with the grouped forward's own candidates, its space the tile its derivation takes and a
    paired product's widest, 128 x 256 at two stages, which spills in every compile (two accumulators of 128
    registers a thread): a launch whose tuner meets a spilling candidate beside one that holds."""

    def candidate(self, ba, config, target):
        return super(Uncounted, self).candidate(ba, config, target)

    def _get_autotuning_configs(self, ba):
        return {ba.heuristics_config, widest(ba.heuristics_config)}


def widest(derived):
    """`derived`'s paired product at its widest, 128 x 256 at two stages (`SpillingFirst`)."""
    columns = derived.grid_block_n * derived.block_n
    return dataclasses.replace(derived, block_n=256, grid_block_n=columns // 256, stages=2)


@lowers("sm90")
def test_a_tuner_times_no_candidate_the_gpus_assembler_refuses():
    """The configs a measuring tuner times (`tuning._candidates`) are those its launch's certification holds
    (`report.certified`, as a run's derivation certifies its config): a paired launch's widest tile, which
    spills in every compile, is never timed nor chosen, where uncertified it was timed as slow as its spills
    and could be picked; the tile its derivation takes, which holds, is."""
    from dew.nn.kernels import tuning
    from dew.nn.kernels.generation import compile_only_gpus

    op = SpillingFirst()
    operands = [jax.ShapeDtypeStruct(shape, dtype) for shape, dtype in (
        ((ROWS, FEATURES), jnp.bfloat16), ((EXPERTS, FEATURES, WIDTH), jnp.bfloat16),
        ((EXPERTS,), jnp.int32))]
    with generation.assumed("sm90"), report.certifying(compile_only_gpus("sm90")[0]), report.refusals():
        bound = op.bind(*operands, paired=jax.ShapeDtypeStruct((EXPERTS, FEATURES, WIDTH), jnp.bfloat16),
                        out_dtype="bfloat16", trans_rhs=False, interpret=False,
                        hooks=Hooks(epilogue=lambda g, u: jax.nn.silu(g) * u))
        derived = bound.heuristics_config
        # The widest is a config a pin may name, which uncertified the tuner timed.
        assert op.pinnable(bound, widest(derived))
        assert tuning._candidates(op, bound) == [derived]


RECONFIGURING_PTX = """
.version 8.8
.target sm_90a
.address_size 64

.visible .entry reconfiguring() .maxntid 384, 1, 1 .maxnreg 168
{
	.reg .b32 %r<2>;
	.reg .pred %p<2>;
	mov.u32 %r1, %tid.x;
	setp.lt.u32 %p1, %r1, 128;
	@%p1 bra PRODUCER;
	setmaxnreg.inc.sync.aligned.u32 232;
	ret;
PRODUCER:
	setmaxnreg.dec.sync.aligned.u32 40;
	ret;
}
"""


def test_a_binary_states_its_threads_and_the_registers_its_roles_reconfigure_to(tmp_path):
    """A kernel of 384 threads whose producer lowers to 40 registers and whose consumers raise to 232
    (`setmaxnreg`), assembled for an H100 by the gate's ptxas (`kernel_compile.assemblers`): its binary
    (`assembled.binary_of`, decoded) states its threads from its `.nv.info` and the two counts from its
    machine code (USETMAXREG), the facts the compile gate reads where no PTX of a kernel was dumped (row
    7105)."""
    from kernel_compile import assemblers

    source = tmp_path / "reconfiguring.ptx"
    source.write_text(RECONFIGURING_PTX)
    cubin = tmp_path / "reconfiguring.cubin"
    subprocess.run([str(assemblers("sm90")[0]), "-arch=sm_90a", str(source), "-o", str(cubin)], check=True)
    binary = assembled.binary_of("reconfiguring", cubin.read_bytes(), decoded=True)
    assert (binary.threads, binary.decreased, binary.raised) == (384, 40, 232), binary

