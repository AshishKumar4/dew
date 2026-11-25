"""The attention template's launches certified before their programs compile (`report.certified`): a
refused backward taking another route; a launch's kernels named alike on every part of a generation and
size, certified once a machine, its stored verdict read where its ptxas build assembles and in a tree edited
outside its kernels; a program's launches compiled beside its trace; a refused launch stating its model;
a spilling ladder certifying its ends; a start running the kernels its launches certified. Split from
tests/test_attention_mosaic_compile.py, whose length passed SLOP009's."""

import dataclasses
import os
import shlex
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from generation_support import COMPILES_ON_CPU
from jax.sharding import Mesh
from kernel_compile import lowered_for
from test_attention_mosaic_compile import _gated_backward, _gated_hooks

from dew.nn.kernels import generation
from dew.nn.kernels.templates import (
    attention as template,
    attention_mosaic,
    attention_mosaic_derivation,
    attention_route,
    report,
    resources,
)

pytestmark = pytest.mark.kernel


@COMPILES_ON_CPU
def test_a_backward_certification_refuses_every_config_of_takes_another_route():
    """A training call whose Mosaic backward certification refuses every config of on a compile-only A100
    (causal bf16 over 512 tokens, four query heads over two key heads, width 128, the accurate backward, a
    query temperature) lowers its public gradient through the route: the route reads certification's
    exhausted verdict as the Mosaic mainloop's refusal and takes another (the base review of 4edd498ac, F3:
    the verdict discarded, the gradient raised Exhausted)."""

    from dew.nn.kernels.generation import compile_only_gpus

    tokens, heads, kv_heads, width = 512, 4, 2, 128
    spans = template.spans(tokens, tokens, 1, causal=True)

    def tempered(s, q_pos, k_pos, temperature):
        return s / temperature

    def loss(query, key, value, temperature):
        hooks = template.Hooks(score_mod=tempered,
                               score_params=(template.Param(temperature, template.Kind.QUERY),))
        out = attention_route.attend(query, key, value, spans, hooks, accurate=True)
        return jnp.sum(out.astype(jnp.float32))

    query = jax.ShapeDtypeStruct((1, tokens, heads, width), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((1, tokens, kv_heads, width), jnp.bfloat16)
    temperature = jax.ShapeDtypeStruct((1, heads, tokens), jnp.float32)
    with report.certifying(compile_only_gpus("sm80")[0]), report.refusals():
        lowered_for(jax.grad(loss, argnums=(0, 1, 2, 3)), (query, key, key, temperature), "sm80")


CERTIFIED_FORWARD = """
import json
import jax
import jax.numpy as jnp
from kernel_compile import lowered_for
from dew.nn.kernels.templates import attention as template, attention_mosaic, report
from dew.nn.kernels.generation import compile_only_gpus

tokens, heads, kv_heads, width = 640, 4, 2, 128
spans = template.spans(tokens, tokens, 1, causal=True)
query = jax.ShapeDtypeStruct((1, tokens, heads, width), jnp.bfloat16)
key = jax.ShapeDtypeStruct((1, tokens, kv_heads, width), jnp.bfloat16)
with report.certifying(compile_only_gpus("sm90")[0]):
    lowered_for(lambda q, k, v: attention_mosaic.forward(q, k, v, spans, None), (query, key, key), "sm90")
print(json.dumps([[found.op, found.stored, {name: module for name, (module, _) in found.kernels.items()}]
                  for found in report.certifications()]))
"""
"""A start's lowering of an H100's attention forward in a process of its own, each launch certified on a
compile-only H100: each certification's op, whether its verdict was read back, and its kernels' modules."""

VERDICT_KEYS = """
import json
import sys
from dew.nn.kernels.templates import report

print(json.dumps({target: report._verdict_key("launch", 1, False, None, target)
                  for target in sys.argv[1:]}))
"""
"""One launch's verdict key (`report._verdict_key`) for each compile target its arguments name, in a process
of its own: what decides it read afresh."""

BUILDS = ("Build cuda_12.9.r12.9/compiler.00000001_0", "Build cuda_12.9.r12.9/compiler.00000002_0")
"""Two builds a ptxas states of itself (`_ptxas_stating`)."""


def _ptxas_stating(directory: Path, build: str | None) -> None:
    """A ptxas in `directory`'s bin that states `build` as its own (`--version`'s last line), its other lines
    the installed ptxas's, and assembles as that one does; with `build` None, one whose `--version` fails."""
    from jax._src.lib import cuda_path

    assert cuda_path is not None, "no installed ptxas to assemble with"
    installed = shlex.quote(os.path.join(cuda_path, "bin", "ptxas"))
    stated = ("exit 1" if build is None else
              f"{installed} --version | sed '$d'; echo {shlex.quote(build)}; exit 0")
    ptxas = directory / "bin" / "ptxas"
    ptxas.parent.mkdir(parents=True, exist_ok=True)
    ptxas.write_text(f'#!/bin/sh\nif [ "$1" = --version ]; then {stated}; fi\nexec {installed} "$@"\n')
    ptxas.chmod(0o755)


@COMPILES_ON_CPU
def test_a_launchs_kernels_are_named_alike_on_every_part_of_one_generation_and_size():
    """An attention forward traced for an H100 and for a GH200, two parts of sm90 of one size (their
    multiprocessors and shared memory), emits the same kernels, and names them alike: a kernel's name
    digests what its emission reads (`report.named`), not the part's name nor its prices, so a step compiled
    ahead on a compile-only H100 (the generation's canonical part) is the program a GH200 runs. The forward's
    launch held the target whole among its static facts, and a GH200's run named all six of its forward
    kernels apart from its build's, every program holding them missed (row 8372, M-MC-27-3f)."""
    from dew.nn.kernels.lowered import kernel_calls

    tokens, heads, kv_heads, width = 640, 4, 2, 128
    spans = template.spans(tokens, tokens, 1, causal=True)
    query = jax.ShapeDtypeStruct((1, tokens, heads, width), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((1, tokens, kv_heads, width), jnp.bfloat16)

    def named(part: str) -> list[tuple[str, str | None]]:
        # Lowered for a compile-only GPU of the generation as `kernel_compile.lowered_for` lowers, within the
        # part's own facts (`assumed` of its target, which `lowered_for`'s generation would replace).
        from jax.experimental.mosaic.gpu import core as mgpu_core
        from jax.sharding import NamedSharding, PartitionSpec

        target = generation.gpu_target(part)
        mesh = Mesh([generation.compile_only_gpus("sm90")[0]], ("x",))
        placed = [jax.ShapeDtypeStruct(spec.shape, spec.dtype, sharding=NamedSharding(mesh, PartitionSpec()))
                  for spec in (query, key, key)]
        with (generation.assumed(target), jax.sharding.use_abstract_mesh(mesh.abstract_mesh),
              mgpu_core.artificial_shared_memory_limit(target.shared_limit)):
            lowered = jax.jit(lambda q, k, v: attention_mosaic.forward(q, k, v, spans, None)).trace(
                *placed).lower()
        # Each kernel by its name and the digest of its launch the name carries (`report.named`).
        return sorted((call.name, call.launch) for call in kernel_calls(lowered, unrun=True))

    h100, gh200 = generation.gpu_target("NVIDIA H100 80GB HBM3"), generation.gpu_target("NVIDIA GH200 480GB")
    assert (h100.generation, h100.multiprocessors, h100.shared_limit) == (
        gh200.generation, gh200.multiprocessors, gh200.shared_limit)
    names = named("NVIDIA H100 80GB HBM3")
    assert names and all(launch for _, launch in names), names
    assert names == named("NVIDIA GH200 480GB"), names


@COMPILES_ON_CPU
def test_a_launch_is_certified_once_a_machine(tmp_path):
    """A launch certified by one process (`report.certified`: compiled alone, its verdict stored with the
    kernel picks) is certified in the next on the same machine by its stored verdict, nothing compiled, the
    same kernels by module: a warm start pays nothing to certify."""
    from process_support import fresh, keeping_picks

    def certified(found: list) -> list:
        return sorted((op, sorted(modules.items())) for op, _, modules in found)

    cold = fresh(CERTIFIED_FORWARD, cache_dir=tmp_path, flags=keeping_picks())
    warm = fresh(CERTIFIED_FORWARD, cache_dir=tmp_path, flags=keeping_picks())
    assert cold and not any(stored for _, stored, _ in cold), cold
    assert all(stored for _, stored, _ in warm), warm
    assert certified(warm) == certified(cold)

@COMPILES_ON_CPU
def test_a_launchs_stored_verdict_is_read_where_its_ptxas_build_assembles_and_not_another(tmp_path):
    """A launch certified by one process is certified by its stored verdict in the next where the same ptxas
    build assembles its kernels, and again where another build does at the same path: the ptxas Mosaic GPU
    names (`_get_assembler`, here `CUDA_HOME`'s, a path holding a parenthesis and a comma) judges a kernel's
    spills and serialized products, so its build keys the verdict (row 10782: the lowered key named none). The
    nvcc wheels named it before, never a system CUDA's ptxas (row 7309)."""
    from process_support import fresh, keeping_picks

    home = tmp_path / "CUDA (13), local"

    def run() -> list:
        return fresh(CERTIFIED_FORWARD, cache_dir=tmp_path / "xla", flags=keeping_picks(),
                     variables={"CUDA_HOME": str(home), "XDG_CACHE_HOME": str(tmp_path / "cache")})

    _ptxas_stating(home, BUILDS[0])
    cold, warm = run(), run()
    _ptxas_stating(home, BUILDS[1])
    reassembled = run()
    assert cold and not any(stored for _, stored, _ in cold), cold
    assert all(stored for _, stored, _ in warm), warm
    assert reassembled and not any(stored for _, stored, _ in reassembled), reassembled


@COMPILES_ON_CPU
def test_a_cuda_targets_verdict_key_names_the_ptxas_jax_gives_xla_and_none_stating_no_build(tmp_path):
    """XLA assembles a Pallas-Triton kernel with the ptxas in the CUDA directory JAX gives each compile
    (`CUDA_ROOT`'s here), which Mosaic GPU's assembler does not search: a CUDA target's verdict key names that
    ptxas's build, so another build at the same path keys a launch afresh, and one stating none (its
    `--version` failing) keys it nowhere, its verdict unstored, where certifying the launch raised (row
    7309)."""
    from process_support import fresh

    root = tmp_path / "cuda"

    def key(build: str | None) -> str | None:
        _ptxas_stating(root, build)
        return fresh(VERDICT_KEYS, "sm90", variables={"CUDA_ROOT": str(root)})["sm90"]

    first, other, again, unstated = (key(build) for build in (*BUILDS, BUILDS[0], None))
    assert first is not None and other is not None and first == again != other
    assert unstated is None


@COMPILES_ON_CPU
def test_a_cuda_targets_verdict_key_names_each_nvjitlink_the_loader_could_open(tmp_path):
    """XLA assembles a Pallas-Triton kernel through the libnvJitLink its stub opens by soname, a system one
    where no wheel provides it: a CUDA target's verdict key names each the process's loader could find (its
    `LD_LIBRARY_PATH`'s here) by the name it is found by and the file that resolves to, so one updated in
    place keys a launch afresh, as does its soname pointed at another build beside it, and the same ones
    key it alike (row 12166: the key read the nvJitLink wheels' versions alone; the review of
    toolkit/nvjitlink-key 13b1e490b: a soname's retarget kept its key)."""
    from process_support import fresh

    directory = tmp_path / "lib"
    directory.mkdir()
    builds = {name: directory / f"libnvJitLink.so.{name}" for name in ("13.0", "13.1")}
    for name, build in builds.items():
        build.write_bytes(f"build {name}".encode())
        os.utime(build, ns=(10**18, 10**18))
    (directory / "libnvJitLink.so").symlink_to(builds["13.0"].name)
    soname = directory / "libnvJitLink.so.13"

    def key(target: str) -> str | None:
        soname.unlink(missing_ok=True)
        soname.symlink_to(builds[target].name)
        return fresh(VERDICT_KEYS, "sm90", variables={"LD_LIBRARY_PATH": str(directory)})["sm90"]

    first, again, retargeted = key("13.0"), key("13.0"), key("13.1")
    builds["13.0"].write_bytes(b"build 13.0, rebuilt")
    updated = key("13.0")
    assert first is not None and first == again
    assert len({first, retargeted, updated}) == 3


def test_a_tpu_targets_verdict_key_reads_no_gpu_assembler():
    """A TPU target's verdict key, on an install whose Mosaic GPU bindings do not import (a TPU-only one),
    names no GPU assembler: reading Mosaic GPU's raised there for every MXU launch certified."""
    from process_support import fresh

    unbound = "import sys\nsys.modules['jax._src.lib.mosaic_gpu'] = None\n" + VERDICT_KEYS
    assert fresh(unbound, "v5e")["v5e"] is not None


EDITED_FORWARD = """
import jax
lowered, compiled = [], []


def heard(event, seconds, *, fun_name="", **_):
    if fun_name.startswith("jit(certified_"):
        if event == "/jax/core/compile/jaxpr_to_mlir_module_duration":
            lowered.append(fun_name)
        elif event == "/jax/core/compile/backend_compile_duration":
            compiled.append(fun_name)


jax.monitoring.register_event_duration_secs_listener(heard)
""" + CERTIFIED_FORWARD.replace("print(json.dumps([[", "print(json.dumps([lowered, compiled, *[[", 1).replace(
    "for found in report.certifications()]))", "for found in report.certifications()]]))", 1)
"""`CERTIFIED_FORWARD`, first naming each certification program it lowered and each it compiled."""

HOOKED_FORWARD = EDITED_FORWARD.replace("import jax\n", "import jax\n\n\ndef capped(s, q_pos, k_pos):\n"
                                        "    return 30.0 * jax.numpy.tanh(s / 30.0)\n", 1).replace(
    "attention_mosaic.forward(q, k, v, spans, None)",
    "attention_mosaic.forward(q, k, v, spans, template.Hooks(score_mod=capped))", 1)
"""`EDITED_FORWARD` of a launch with a score hook (a soft cap)."""


@COMPILES_ON_CPU
def test_a_launch_is_certified_by_its_kernels_verdict_in_a_tree_edited_outside_them(tmp_path):
    """A launch certified by one process is certified in the next on the same machine by its stored verdict
    though Dew's code changed: outside the launch's code (`report.launch_code`: a trainer's), read before it
    traces, nothing lowered; in its op's module, outside its kernels (`report._lowered_verdict_key`), the
    launch lowered, its kernels the same, and their verdict read, compiling none. Keyed by Dew's whole code
    alone, every commit compiled every launch of a start again (row 8650: 23 of an edited tree's 51 seconds
    certifying a hybrid decoder's step ahead, FS-8650-SPLIT), and lowered each (row 10743). A launch with a
    hook, which its identity traces at one tile alone, keys on all of Dew's code: an edit anywhere lowers it
    again."""
    import shutil

    from process_support import REPO_ROOT, fresh, keeping_picks

    tree = tmp_path / "src"
    for name in ("dew", "_dew_torch"):
        shutil.copytree(REPO_ROOT / "src" / name, tree / name, ignore=shutil.ignore_patterns("__pycache__"))

    def run(script: str = EDITED_FORWARD) -> list:
        return fresh(script, cache_dir=tmp_path / "xla", flags=keeping_picks(), source=tree,
                     variables={"XDG_CACHE_HOME": str(tmp_path / "cache")})

    def edited(*parts: str) -> None:
        path = tree.joinpath("dew", *parts)
        path.write_text(path.read_text() + "\n# edited\n")

    def certified(found: list) -> list:
        return sorted((op, sorted(modules.items())) for op, _, modules in found)

    _, cold_compiled, *cold = run()
    assert cold and cold_compiled and not any(stored for _, stored, _ in cold), cold
    assert run(HOOKED_FORWARD)[1]
    edited("training", "rungs.py")
    lowered, compiled, *elsewhere = run()
    assert all(stored for _, stored, _ in elsewhere) and certified(elsewhere) == certified(cold), elsewhere
    assert (lowered, compiled) == ([], [])
    lowered, compiled, *hooked = run(HOOKED_FORWARD)
    assert hooked and all(stored for _, stored, _ in hooked) and lowered and not compiled, (lowered, hooked)
    edited("nn", "kernels", "templates", "attention_mosaic.py")
    lowered, compiled, *op_edited = run()
    assert all(stored for _, stored, _ in op_edited) and certified(op_edited) == certified(cold), op_edited
    assert lowered and not compiled, (lowered, compiled)


CERTIFIED_PROGRAM = """
import json
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from dew.nn.attention import scaled_dot_product_attention
from dew.nn.kernels import generation
from dew.nn.kernels.generation import compile_only_gpus
from dew.nn.kernels.options import owned_program
from dew.nn.kernels.templates import report

tokens, heads, kv_heads, width = 1600, 4, 2, 128
device = compile_only_gpus("sm90")[0]
placed = NamedSharding(Mesh([device], ("x",)), PartitionSpec())
query = jax.ShapeDtypeStruct((1, tokens, heads, width), jnp.bfloat16, sharding=placed)
key = jax.ShapeDtypeStruct((1, tokens, kv_heads, width), jnp.bfloat16, sharding=placed)

def loss(query, key, value):
    attended = scaled_dot_product_attention(query, key, value, causal=True, implementation="template")
    return jnp.sum(attended.astype(jnp.float32))

import threading

compiled_on = []

def heard(event, seconds, **_):
    if event == "/jax/core/compile/backend_compile_duration":
        compiled_on.append(threading.current_thread().name)

jax.monitoring.register_event_duration_secs_listener(heard)
try:
    with generation.assumed("sm90"), report.certifying(device), report.refusals():
        owned_program(jax.grad(loss, argnums=(0, 1, 2))).compiled(query, key, key)
finally:
    jax.monitoring.unregister_event_duration_listener(heard)
print(json.dumps({"certs": [[found.op, found.stored, bool(found.refused),
                             {name: module for name, (module, _) in found.kernels.items()}]
                            for found in report.certifications()],
                  "compiled_on": compiled_on}))
"""
"""A training step compiled through Dew's compile point (`OuterProgram.compiled`) in a process of its own,
each attention launch certified on a compile-only H100."""


@COMPILES_ON_CPU
def test_a_programs_launches_compile_beside_its_trace(tmp_path):
    """A training step compiled through Dew's compile point (`OuterProgram.compiled`) certifies its
    attention launches beside its trace on `dew-certify` threads (`report.certifying_aside`) on a cold
    isolated store and reads stored verdicts on a warm repeat without refusing any launch (rows 10802,
    10834)."""
    from process_support import fresh, keeping_picks

    def certified(found: list) -> list:
        return sorted((op, sorted(modules.items())) for op, _, _, modules in found)

    cold = fresh(CERTIFIED_PROGRAM, cache_dir=tmp_path, flags=keeping_picks())
    warm = fresh(CERTIFIED_PROGRAM, cache_dir=tmp_path, flags=keeping_picks())
    cold_certs, warm_certs = cold["certs"], warm["certs"]
    assert cold_certs and not any(refused for _, _, refused, _ in cold_certs), cold_certs
    assert not any(stored for _, stored, _, _ in cold_certs), cold_certs
    aside = [name for name in cold["compiled_on"] if name.startswith("dew-certify")]
    assert len(aside) == len(cold_certs) > 0, cold["compiled_on"]
    assert warm_certs and all(stored for _, stored, _, _ in warm_certs), warm_certs
    assert not any(refused for _, _, refused, _ in warm_certs), warm_certs
    assert certified(warm_certs) == certified(cold_certs)


@COMPILES_ON_CPU
def test_a_refused_launch_states_its_model_beside_the_assemblers_registers():
    """A launch certification refuses records each refused kernel with its launch and config, ptxas's
    registers and stack frame beside the registers its launch's model stated, and steps it down where the
    derivation offers another: the accurate backward 64 wide on an H100, configured at a 128-row query
    block, which its dQ spills at."""
    from dew.nn.kernels.generation import compile_only_gpus

    op, ba, derived = _gated_backward("sm90", 64, accurate=True)

    assert isinstance(derived, attention_mosaic.BackwardTiles), derived
    # Its tile at 128 query rows, which its dQ spills at (FS-DECISION-sm90).
    config = dataclasses.replace(derived, block_q=128)
    with generation.assumed("sm90"), report.certifying(compile_only_gpus("sm90")[0]), report.refusals():
        spilling = report.certified(op, ba, config)
        assert spilling, config
        report.missed(op, ba, config, spilling, stepped=True)
        missed = [miss for miss in report.misses()
                  if miss.op == type(op).__name__ and miss.config == config]
        budget = op.register_budget(config)
    assert len(missed) == len(spilling), (missed, spilling)
    refused_by_ptxas = [(miss.spilled or 0) > 0 or (miss.registers or 0) > budget or bool(miss.serialized)
                        for miss in missed]
    assert all(refused_by_ptxas), (missed, budget)
    assert all(miss.modelled is not None and miss.stepped for miss in missed), missed


@COMPILES_ON_CPU
def test_a_ladder_that_spills_throughout_certifies_its_ends_alone():
    """The accurate backward under an alibi hook 64 wide on an H100, whose dQ spills at every query block the
    derivation offers (row 7938): certification refuses each of them, as it would certifying every rung,
    having certified fewer of its configs than it refused, the rungs between a regime's two refused ends
    refused by inference (`tuning.KernelOp._spanned`)."""
    from dew.nn.kernels.generation import compile_only_gpus

    op, ba, derived = _gated_backward("sm90", 64, accurate=True, hooks="alibi")

    assert isinstance(derived, attention_mosaic.BackwardTiles), derived
    assert derived is not None
    with generation.assumed("sm90"), report.certifying(compile_only_gpus("sm90")[0]), report.refusals():
        before = len(report.certifications())
        taken = op._certified(ba, derived)
        certified = {certification.config for certification in report.certifications()[before:]
                     if certification.op == type(op).__name__}
        rungs = tuple(dataclasses.replace(derived, block_q=block) for block in (256, 192, 128, 64)
                      if block <= derived.block_q)
        refused = {rung for rung in rungs if report.refused(op, ba, rung, generation.gpu_target())}
    assert taken is None or taken not in refused, taken
    assert refused == set(rungs), (rungs, refused)
    assert len(certified & set(rungs)) < len(rungs), (certified, rungs)


@COMPILES_ON_CPU
def test_a_ladder_holds_no_rung_its_assembler_caps_otherwise():
    """The accurate backward under an alibi hook 64 wide on an H100, at a 128-row query block: its dQ's
    program moves registers there and not at 64 rows, so its kernels assemble under other caps (248 against
    255), and its ladder (`MosaicBackward.ladder`, whose rungs a refusal infers refused) holds no rung whose
    launches' stated caps or register movement differ from its own (row 8467)."""
    op, ba, derived = _gated_backward("sm90", 64, accurate=True, hooks="alibi")
    assert isinstance(derived, attention_mosaic.BackwardTiles), derived
    with generation.assumed("sm90"):
        config = dataclasses.replace(derived, block_q=128)
        dtype = ba.arguments["query"].dtype

        def stated(tiles):
            dq, dkv = op._launch_registers(ba, tiles)
            launches = ((tiles.dkv, dkv),) if tiles.fused else ((tiles, dq), (tiles.dkv, dkv))
            return [(model.cap(1), model.moves) for model in (
                resources.register_model(attention_mosaic.bundle_of(each, dtype), each.consumers, registers,
                                         target=generation.gpu_target())
                for each, registers in launches)]

        assert stated(dataclasses.replace(config, block_q=64)) != stated(config)
        assert all(stated(rung) == stated(config) for rung in op.ladder(ba, config)), op.ladder(ba, config)


@COMPILES_ON_CPU
def test_a_fewer_rows_ladder_walks_the_query_blocks_its_derivation_walks():
    """An A100's backward at mma.sync's fewer rows (32 a consumer, `fewer_rows`) from a 128-row query block:
    its ladder (`MosaicBackward.ladder`) offers the query blocks its derivation walks at those rows
    (`_sides` on the layout it derives on, not `bundle_of`'s, whose warps are already arranged for 32
    rows), each one its launch takes within the config's cap and residency (row 8487)."""
    from dew.nn.kernels.templates.attention_mosaic_program import fewer_rows

    op, ba, derived = _gated_backward("sm80", 64, accurate=True)

    assert isinstance(derived, attention_mosaic.BackwardTiles), derived
    with generation.assumed("sm80"):
        target = generation.gpu_target()
        query = ba.arguments["query"]
        layout = op._named(target.generation, query.dtype, query.shape[1])[derived.family]
        assert 32 in fewer_rows(layout), fewer_rows(layout)
        config = dataclasses.replace(derived, block_q=128, block_k=32)
        sides = attention_mosaic_derivation._sides(32, layout, query.dtype, query.shape[1])
        walked = {block for block in sides
                  if block < 128 and op.takes(ba, dataclasses.replace(config, block_q=block), target)
                  and op._regime(ba, dataclasses.replace(config, block_q=block), target)
                  == op._regime(ba, config, target)}
        ladder = {rung.block_q for rung in op.ladder(ba, config)
                  if isinstance(rung, attention_mosaic.BackwardTiles)}
    assert walked and ladder == walked, (ladder, walked)


@COMPILES_ON_CPU
@pytest.mark.parametrize("hooks", ["none", "temperature"])
def test_a_start_runs_the_kernels_its_launches_certified(hooks):
    """A launch's derived config certified as it traces (`report.certified`: its kernels compiled alone as
    the GPU's compiler assembles them, in certification's own placement) on a compile-only B200, and then
    compiled within its program: every Mosaic GPU kernel the program runs is one certification compiled,
    module for module (its kernel hash), which Mosaic compiles once a process, so certification adds a start
    no second assembly and its verdict is the program's kernel's. Under a score hook too, whose traced digest
    names the kernels (row 8532: certification's placement named them apart, and ptxas assembled the two
    otherwise identical modules differently, 8 bytes spilled in the program against none certified)."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    tokens, heads, kv_heads, width = 1600, 4, 2, 128
    spans = template.spans(tokens, tokens, 1, causal=True)
    query = jax.ShapeDtypeStruct((1, tokens, heads, width), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((1, tokens, kv_heads, width), jnp.bfloat16)
    hooked = _gated_hooks(hooks, tokens, width)

    def forward(query, key, value):
        return attention_mosaic.forward(query, key, value, spans, hooked)

    found = compile_mosaic_kernels([KernelCase("a start's forward", lambda: (forward, (query, key, key)))],
                                   "sm100")
    assert mosaic_failures(found) == []
    record, = found
    run = record["modules"]
    assert run and {name: record["certified"].get(name) for name in run} == run
