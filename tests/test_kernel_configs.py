"""CPU first filters for the kernels: each of the fla port's declared configs, and the delta rule's and
the short conv's kernels at their training shapes, compiles for its GPU within the block's shared
memory, and the port's dots refuse mixed dtypes. None runs a kernel, so the GPU tier, whose kernel tests
are the verdict, runs none of them: the file carries no `kernel` mark."""

import tempfile
from pathlib import Path

import fla_gdn_support as support
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tokamax
from fla_gdn_support import CONV_SHAPES, conv_config_cases, conv_training_cases, kda_training_cases
from generation_support import lowers
from kernel_compile import compile_kernels, failures

from dew.nn.kernels import chunked_launch, fla_gdn, fla_kda, generation, recurrence, short_conv, tuning

LAUNCHES = {launch for module in (fla_gdn, fla_kda, recurrence) for launch in vars(module).values()
            if isinstance(launch, chunked_launch._Launch)}
"""Every launch of the two ports and the recurrence template's chunked stages."""


def test_the_compiled_shapes_reach_the_widest_keys_and_values_the_port_takes():
    shapes = support.COMPILED_SHAPES.values()
    assert max(shape["width"] for shape in shapes) == fla_gdn.WIDEST_KEYS == 256
    assert max(shape["columns"] for shape in shapes) == fla_gdn.WIDEST == 128


@pytest.mark.parametrize("target, wide", [("sm80", True), ("sm90", True), ("sm86", False), ("sm89", False),
                                          ("sm120", False), ("sm100", True), ("sm110", False), ("cpu", True)])
def test_a_gpu_takes_an_a100s_picks_where_its_block_holds_as_much_shared_memory(target, wide):
    """An A100's picks on every GPU whose block holds as much shared memory, the 99 KB ones elsewhere,
    on a GPU the targets do not list (sm110) too, and an A100's where a CPU interprets."""
    with generation.assumed(target):
        assert chunked_launch._wide() is wide


@pytest.mark.parametrize("target", [*generation.TARGETS, "sm110", "cpu"])
def test_each_targets_space_holds_its_picks(target):
    """On every target `generation.TARGETS` lists, a GPU it does not and a CPU, every kernel's 16-bit
    space at the target's width holds its unmeasured pick, Kimi delta attention's own kernels' too, and
    fp32 has a pick for every kernel."""
    with generation.assumed(target):
        wide = chunked_launch._wide()
    for launch in LAUNCHES:
        assert launch.PICKS["bfloat16", wide] in launch.SPACES[wide], launch.NAME
        assert {"bfloat16", "float32"} <= {dtype for dtype, _ in launch.PICKS}, launch.NAME


def rule_launches(module) -> tuple:
    """The launches the rule of `module` (`fla_gdn`, `fla_kda`) makes forward and backward, as the
    tuner reads them from its lowering (`tuning.launches`)."""
    per_key = module is fla_kda
    q = jax.ShapeDtypeStruct((1, 64, 1, 64), jnp.bfloat16)
    v = jax.ShapeDtypeStruct((1, 64, 2, 64), jnp.bfloat16)
    g = jax.ShapeDtypeStruct((1, 64, 2, *(64,) * per_key), jnp.float32)
    h0 = jax.ShapeDtypeStruct((1, 2, 64, 64), jnp.float32)
    port = fla_kda.chunk_kda if per_key else fla_gdn.chunk_gated_delta_rule

    def rule(*args):
        return port(*args, None, 0.125, 64, interpret=True)

    def both(q, k, v, g, beta, h0, do):
        # The output returned, so the forward's every launch is live: a dead one is DCE'd (jax 0.12).
        (out, final), pull = jax.vjp(rule, q, k, v, g, beta, h0)
        return out, pull((do, jnp.zeros_like(final)))

    with tuning.launches():
        beta = g.update(shape=g.shape[:3], dtype=jnp.bfloat16)
        return tokamax.autotuning.get_bound_args(jax.jit(both), q, q, v, g, beta, h0, v)


@pytest.mark.parametrize("module", [fla_gdn, fla_kda])
def test_the_tuner_finds_each_of_the_rules_launches_in_its_lowering(module):
    """The rule's forward and backward, traced as the tuner reads a program (`tuning.launches`): each
    of its kernels is a launch tokamax describes, its arguments serializable into its key; the gated
    delta rule's eight and the recurrence template's decay prefix, and Kimi delta attention's seven with
    the prefix and the memory kernels it shares."""
    shared = {recurrence.ChunkLocalCumsum, *((fla_gdn.ChunkFwdH, fla_gdn.ChunkBwdDhu) if module is fla_kda
                                             else ())}
    assert {type(ba.op) for ba in rule_launches(module)} == shared | {
        type(op) for op in vars(module).values() if isinstance(op, chunked_launch._Launch)}


def test_a_launch_clips_its_blocks_to_the_operands():
    """Blocks wider than a call's keys or values shrink to them: on an A100, whose picks block 64 values
    or more, the heuristic and every config the tuner measures fit a call's 64-wide keys and 32-wide
    values, the heuristic among them."""
    q = jax.ShapeDtypeStruct((1, 64, 1, 64), jnp.bfloat16)
    v = jax.ShapeDtypeStruct((1, 64, 1, 32), jnp.bfloat16)
    h = jax.ShapeDtypeStruct((1, 1, 64, 32), jnp.bfloat16)
    g = jax.ShapeDtypeStruct((1, 64, 1), jnp.float32)
    spans = chunked_launch.sequences(1, 64)
    ba = fla_gdn._CHUNK_FWD_O.bind(q, q, v, h, g, *spans, scale=1.0, chunk=64, interpret=True)
    with generation.assumed("sm80"):
        heuristic = fla_gdn._CHUNK_FWD_O._get_heuristics_config(ba)
        space = fla_gdn._CHUNK_FWD_O._get_autotuning_configs(ba)
    assert heuristic in space
    for tiles in space:
        assert tiles.bk <= q.shape[-1] and tiles.bv <= v.shape[-1], tiles


@pytest.mark.parametrize("target", ["sm80", "sm120"])
def test_the_kernel_cases_lower_each_kernel_on_abstract_operands(target):
    """The compile gate's cases (`kernel_cases`) hold each kernel the gated delta rule launches forward
    and backward, at both dtypes at every compiled shape the dtype takes (fp32 keys at most
    `fla_gdn.WIDEST` wide), the decay prefix (`recurrence.cumulated`) reversed too, each once, on
    abstract operands of that shape lowered as on the GPU."""
    cases = list(support.kernel_cases(target))
    names = [case.name.split("/") for case in cases]
    assert len({tuple(name) for name in names}) == len(names)
    assert all(name[0] == target for name in names)
    covered = {(kernel, dtype, shape, *reverse) for _, kernel, dtype, _, shape, *reverse in names}
    launched = {ba.op.NAME for ba in rule_launches(fla_gdn)}
    reversals = {kernel: [(), ("reverse",)] if kernel == "cumulated" else [()]
                 for kernel in launched}
    assert covered == {(kernel, dtype, shape, *reverse)
                       for kernel in launched for dtype in ("bfloat16", "float32")
                       for shape, geometry in support.COMPILED_SHAPES.items()
                       if dtype == "bfloat16" or geometry["width"] <= fla_gdn.WIDEST
                       for reverse in reversals[kernel]}
    by_name = dict(zip((case.name for case in cases), cases, strict=True))
    for name, case in by_name.items():
        fn, args = case.call()
        geometry = support.COMPILED_SHAPES[name.split("/")[4]]
        assert fn.keywords["interpret"] is False
        assert args and all(isinstance(arg, jax.ShapeDtypeStruct) for arg in args)
        assert args[0].shape[:2] == (geometry["batch"], geometry["length"])


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("left,right", [(jnp.bfloat16, jnp.float32), (jnp.float32, jnp.bfloat16),
                                       (jnp.float16, jnp.bfloat16)])
def test_the_shared_dot_rejects_mismatches_before_interpret_widening(exact, left, right):
    a, b = jnp.ones((2, 2), left), jnp.ones((2, 2), right)
    with pytest.raises(TypeError):
        fla_gdn._dot(a, b, exact=exact)


def test_the_shared_dot_preserves_narrow_products_with_fp32_accumulation():
    a = jnp.asarray([[1.0, 0.5], [-0.25, 2.0]], jnp.bfloat16)
    got = fla_gdn._dot(a, a.T, exact=True)
    assert got.dtype == jnp.float32
    expected = a.astype(jnp.float32) @ a.T.astype(jnp.float32)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


@pytest.mark.parametrize("target", ["sm80", "sm89", "sm120"])
def test_every_declared_kernel_config_compiles_on_tritons_compiler_within_its_block(target):
    """Each declared config at each declared shape, lowered as on the GPU and compiled by XLA's CUDA backend
    for a compile-only device on this CPU (tests/kernel_compile.py): it compiles, to no more shared memory
    than its GPU's block holds. A kernel that does not compile (a two-result inline asm, a mixed-dtype
    dot) fails here and never reaches a GPU."""
    found = compile_kernels(support.kernel_cases(target), target)
    assert found, "the compile gate must not be vacuous"
    assert not failures(found), "\n".join(failures(found))


@pytest.mark.parametrize("target", ["sm80", "sm89", "sm120"])
def test_kimi_delta_attentions_kernels_compile_on_tritons_compiler_within_each_gpus_block(target):
    """Every kernel of the port's forward and backward at the training shape, lowered as on the GPU and
    compiled by XLA's CUDA backend for a compile-only device on this CPU (tests/kernel_compile.py), within
    the compile bound and the block's shared memory."""
    found = compile_kernels(kda_training_cases(), target)
    assert len(found) == 2
    assert not failures(found), "\n".join(failures(found))


@pytest.mark.parametrize("target", ["sm80", "sm89", "sm120"])
def test_the_short_convs_kernels_compile_on_tritons_compiler_within_each_gpus_block(target):
    """Each of the short conv's training cases lowered as on the GPU and compiled by XLA's CUDA backend for a
    compile-only device on this CPU (tests/kernel_compile.py), within the compile bound and the block's
    shared memory."""
    found = compile_kernels(conv_training_cases(), target)
    assert len(found) == 2 * len(CONV_SHAPES) + 2
    assert not failures(found), "\n".join(failures(found))


@pytest.mark.parametrize("target", ["sm80", "sm89", "sm120"])
def test_every_short_conv_config_compiles_on_tritons_compiler_within_each_gpus_block(target):
    """Each config the tuner may give either launch of the short conv (`short_conv.TILES`,
    `short_conv.WALKS`), pinned at the 9B's packed delta conv, compiled for each GPU within the compile
    bound and the block's shared memory."""
    found = compile_kernels(conv_config_cases(), target)
    assert len(found) == len(short_conv.TILES) + len(short_conv.WALKS)
    assert not failures(found), "\n".join(failures(found))


@lowers("sm90")
def test_the_cooperative_grouped_matmul_compiles_for_hopper_within_its_resources():
    """The GEMM template's Mosaic GPU grouped matmul, lone and cooperative, compiled by Mosaic for a
    compile-only H100 on this CPU through ptxas (tests/kernel_compile.py): within the multiprocessor's
    shared memory, no register spilled. The resource gate every Mosaic GPU mainloop takes."""
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    from dew.nn.kernels.templates.gemm import mosaic_gpu

    def case(tiles):
        def call():
            def gmm(lhs, rhs, sizes):
                hopper = generation.gpu_target("sm90")
                return mosaic_gpu.gmm(lhs, rhs, sizes, tiles=tiles, transpose_rhs=False,
                                      out_dtype=jnp.bfloat16, multiprocessors=hopper.multiprocessors)

            return gmm, (jax.ShapeDtypeStruct((1024, 512), jnp.bfloat16),
                         jax.ShapeDtypeStruct((4, 512, 1024), jnp.bfloat16),
                         jax.ShapeDtypeStruct((4,), jnp.int32))

        return support.KernelCase(f"gmm {tiles}", call)

    found = compile_mosaic_kernels([case(mosaic_gpu.Tiles(128, 128, 64, 3, 1)),
                                    case(mosaic_gpu.Tiles(128, 256, 64, 3, 1, warpgroups=2))], "sm90")
    assert len(found) == 2 and all(record.get("kernels") for record in found), found
    assert not mosaic_failures(found), "\n".join(mosaic_failures(found))


def test_the_resource_gate_refuses_a_spilling_mosaic_kernel():
    """A record the gate reads (`mosaic_failures`): a kernel past its shared memory, its tensor memory or
    spilling fails; within all three, none."""
    from kernel_compile import mosaic_failures, mosaic_kernels

    within = {"kernel": "k", "unread": [], "shared": 1024, "tmem": 256, "registers": 168, "spilled": 0}
    record = {"name": "case", "limit": 2048, "shared": 0, "tmem_limit": 512, "kernels": [within]}
    assert not mosaic_failures([record])
    for past in ({"shared": 4096}, {"tmem": 513}, {"spilled": 40}):
        assert mosaic_failures([{**record, "kernels": [{**within, **past}]}])
    # The program's own refusal of a block past shared memory fails it whatever its Mosaic kernels say.
    assert mosaic_failures([{**record, "shared": 4096}])
    # Either report without the other, or a report without a field, is read as missing, not as within.
    complete = ("smem_scratch_bytes=1024 tmem_scratch_cols=0, tmem_collective_scratch_cols=0",
                "Used 168 registers\n0 bytes spill stores, 0 bytes spill loads")
    with tempfile.TemporaryDirectory() as directory:
        Path(directory, "a.resources.txt").write_text(complete[0])
        Path(directory, "a.ptxas").write_text(complete[1])
        spilling = "Used 255 registers\n40 bytes spill stores, 40 bytes spill loads"
        Path(directory, "b.ptxas").write_text(spilling)
        Path(directory, "c.resources.txt").write_text("smem_scratch_bytes=1024")
        a, b, c = mosaic_kernels(directory)
        assert not a["unread"]
        assert set(b["unread"]) == {"shared bytes", "tensor memory columns"}
        assert set(c["unread"]) == {"tensor memory columns", "registers", "spills"}
        assert len(mosaic_failures([{**record, "kernels": [a, b, c]}])) == 2


def test_splash_compiles_for_a_v5e_with_libtpu():
    """Dew's splash path, forward and backward, compiled by libtpu for a compile-only v5e on this CPU
    (tests/kernel_compile.py): within the v5e's scoped VMEM, as the blocks splash takes are sized
    (`dew.nn.attention.splash_block_sizes`)."""
    from kernel_compile import compile_tpu_kernels

    from dew.nn.attention import scaled_dot_product_attention

    def call():
        def step(query, key, value):
            def loss(q, k, v):
                return jnp.sum(scaled_dot_product_attention(q, k, v, implementation="tpu", causal=True)
                               .astype(jnp.float32))
            return jax.grad(loss, argnums=(0, 1, 2))(query, key, value)

        shape = jax.ShapeDtypeStruct((1, 2048, 4, 128), jnp.float32)
        return step, (shape, shape, shape)

    found = compile_tpu_kernels([support.KernelCase("splash fp32", call)], "v5e")
    assert not [record for record in found if "error" in record], found
