"""The recurrences on the GPU (G14, the owner's rule: no unit tests on GPUs): each native recurrence route's
cases the CPU does not interpret, run in one process (`gpu_flow`), each cell its own file's property:

- The template finite and on par where chunked forms break, every form, regime and product
  (test_recurrence_template_stability).
- The portable backward as a GPU compiles it, and the SSD port's forward and gradient, finite and on par
  under every regime (test_recurrence_stability).
- A Mamba layer under the template's selective scan, packed and from a held state, as exact as under the
  portable scan, on an H100 (test_selective_template).
- A two-role walk's training on an H100, its two consumers a launch no flow's widths reach
  (test_recurrence_mosaic_gradient).
- The template over a forked call, its walks taking each document's memory from the call's table and storing
  the finals later documents take (test_recurrence_chunk), and, in a user's process, its fp32 forward at the
  default precision on each GPU's instruction family, its training on an H100's (test_recurrence_mosaic, its
  gradient's).
- Each recurrence layer trained on its port's kernels, natively (`PORT_KERNELS`): a gated delta net on the
  FLA GDN kernels and a Kimi delta attention layer on FLA KDA's chunk_kda, packed rows, forward and every
  gradient on par from float64 with the portable rule's; their training step the same bit for bit each
  run, on the port and on the template; and the port on par with fla's own in fp32 and mixed dtypes
  (test_fla_gdn), fla run in the reference interpreter DEW_FLA_PYTHON names, which a GPU job provides to a
  file naming it.
- The decode launches as a server runs them (test_linear_attention): the step kernel leaving an idle row's
  state and output alone, bit for bit; the fused step's bits the same however its launches are timed; and
  a layer choosing the fused step decoding over two GPUs, its rows or its heads split, as on one device.

- The check-1 matrices of the recurrence's choices (tests/kernel_matrix.py): the chunked op on each kernel
  against its form's port (recurrence_matrix), the short conv against XLA's (short_conv_matrix), the
  selective scan against recorded mamba_ssm (selective_matrix), and the decode tokens on each decode kernel
  against the portable step (delta_decode_matrix), each over the combinations of the axes a GPU runs
  natively, its form, kernel and precision, and the cases a red found on a GPU (`kernel_matrix.native_cells`).

The Mosaic mainloop's forward and gradient, and each recurrence's chunked op, short conv and decode on its
default kernels at a model's widths, are the model flows' (tests/test_gpu_flows.py, row 5461's covering
set); their shapes and hooks past these are the CPU's, interpreted and through the compile gate.
"""

import functools

import gpu_flow
import jax
import jax.numpy as jnp
import kernel_matrix
import numpy as np
import pytest
import test_fla_gdn as fla_gdn
import test_mamba_ssd as mamba_ssd
import test_recurrence_chunk as chunked
import test_recurrence_mosaic as forward
import test_recurrence_mosaic_gradient as gradient
import test_recurrence_stability as stability
import test_recurrence_template_stability as template_stability
import test_selective_template as selective_template
from delta_rule_support import fused_step
from gpu_flow import Cell, covering, run_cells, when
from jax.sharding import NamedSharding, PartitionSpec
from model_support import decode_through
from reference_error import FEWEST_SEEDS, as_given, assert_as_exact, with_gradients
from test_linear_attention import decode_operands, fused_case, fused_inputs

from dew.batch import coordinates
from dew.nn.inputs import AttentionMetadata, MixerCall
from dew.nn.kda import KimiDeltaAttention
from dew.nn.kernels import generation
from dew.nn.linear import GatedDeltaNet, decode_gated_delta_rule
from dew.training import MeshSpec

pytestmark = pytest.mark.gpu(reason="each recurrence route's native kernel no model flow reaches: the ports, "
                                    "the template at its edges")


PORT_KERNELS = {
    "fla": ("fla_gdn: kkt_solve, recompute_w_u, chunk_fwd_h, chunk_fwd_o, chunk_bwd_dv_local, chunk_bwd_dhu, "
            "chunk_bwd_dqkwg, prepare_wy_repr_bwd"),
    "kda": ("fla_kda: intra_rows, intra_solve, recompute_w_u, chunk_fwd_o, bwd_dav, bwd_wy_dqkg, bwd_intra; "
            "fla_gdn: chunk_fwd_h, chunk_bwd_dhu"),
}
"""The port kernels each layer's training cell runs natively, by the layer's port (`trains_on_its_port`)."""

ROWS, TOKENS, WIDTH = 2, 256, 256
"""A layer cell's rows, their tokens (four 64-token chunks) and the layers' model width."""


def gated_delta_net():
    return GatedDeltaNet(emb_features=WIDTH, num_k_heads=2, num_v_heads=4, head_k_dim=128, head_v_dim=128,
                         dtype=jnp.bfloat16)


def kimi_delta_attention():
    return KimiDeltaAttention(emb_features=WIDTH, num_heads=2, head_dim=128, dtype=jnp.bfloat16)


def named(fn, **kernels: str):
    """`fn` with the recurrence's selectors named to `kernels` (`generation.chosen`)."""
    def run(*args):
        with generation.chosen(**kernels):
            return fn(*args)
    return run


def trains_on_its_port(make):
    """A layer over two rows of two packed documents each, its forward and the gradients of its input and
    every parameter under a drawn cotangent on its port (`PORT_KERNELS`, named), on par from float64 with
    the portable rule's (check 1, the layer in float64 the truth)."""
    layer = make()
    packed = MixerCall(AttentionMetadata(coordinates=coordinates(
        segments=jnp.asarray([[1] * 150 + [2] * 106, [1] * 90 + [2] * 166]), max_segments=2)))

    def apply(params, x):
        return layer.apply(params, x, packed)

    def draw(seed):
        x = jax.random.normal(jax.random.key(seed), (ROWS, TOKENS, WIDTH)).astype(jnp.bfloat16)
        params = named(layer.init, gated_delta_rule="xla")(jax.random.key(1000 + seed), x)
        cotangent = jax.random.normal(jax.random.key(2000 + seed), x.shape).astype(jnp.bfloat16)
        return params, x, cotangent

    def truth(params, x, cotangent):
        wide = layer.clone(dtype=jnp.float64)
        return with_gradients(lambda params, x: wide.apply(params, x, packed))(params, x, cotangent)

    assert_as_exact(named(with_gradients(apply), gated_delta_rule="fla"),
                    named(with_gradients(apply), gated_delta_rule="xla"), draw,
                    f"{type(layer).__name__} trained on its port", seeds=FEWEST_SEEDS, truth_fn=truth)


def repeats_bit_for_bit(make, kernel: str):
    """A layer's training step on `kernel`, the same program on the same inputs, run twice, compiled afresh
    the second time: every gradient the same bit for bit (the kernels sum in a fixed order)."""
    layer = make()
    x = jax.random.normal(jax.random.key(5), (ROWS, TOKENS, WIDTH)).astype(jnp.bfloat16)
    params = named(layer.init, gated_delta_rule="xla")(jax.random.key(6), x)
    step = jax.jit(named(jax.grad(lambda params, x: jnp.sum(layer.apply(params, x).astype(jnp.float32) ** 2),
                                  argnums=(0, 1)), gated_delta_rule=kernel))
    gpu_flow.repeats_bit_for_bit(step, params, x)


def the_decode_kernel_leaves_an_idle_row_alone():
    """A serving step holds every slot, drawing or not: a row the step kernel (`decode_gated_delta_rule`)
    marks idle, at Qwen3.5-0.8B's 16 heads 128 wide, keeps its state bit for bit, unread and unwritten, and
    outputs zeros; the drawing rows step to the bits they take with every row active."""
    query, key, value, g, beta, state = decode_operands(np.random.default_rng(1), 8, 16, 128)
    held = np.asarray(state)
    active = jnp.asarray([True, False, True, True, False, False, True, False])
    every, every_state = jax.jit(decode_gated_delta_rule)(query, key, value, g, beta, state)
    got, got_state = jax.jit(decode_gated_delta_rule)(query, key, value, g, beta, state, active)
    drawing = np.asarray(active)
    np.testing.assert_array_equal(np.asarray(got)[drawing], np.asarray(every)[drawing])
    np.testing.assert_array_equal(np.asarray(got_state)[drawing], np.asarray(every_state)[drawing])
    np.testing.assert_array_equal(np.asarray(got)[~drawing], 0)
    np.testing.assert_array_equal(np.asarray(got_state)[~drawing], held[~drawing])


def the_fused_decode_step_gives_the_same_bits_however_its_launches_are_timed():
    """The fused decode step (`fused_step`) at Qwen3.5-0.8B's widths, a bf16 state, run 20 times, each run
    beside a large matmul that holds most of the GPU or alone, gives the same output, history and state bits
    every run: no value head reads a history or state another program is writing, whatever order they run
    in."""
    layer, variables, cache, token = fused_case(16, 128, 4, jnp.bfloat16, jnp.bfloat16)
    operands = fused_inputs(layer, variables, cache, token)
    load = jnp.ones((8192, 8192), jnp.bfloat16)
    busy = jax.jit(lambda a: a @ a)
    run = jax.jit(functools.partial(fused_step, epsilon=layer.norm_eps, activation=layer.gate_activation,
                                    dtype=jnp.bfloat16))
    first = jax.tree.map(np.asarray, run(*operands))
    for repeat in range(20):
        if repeat % 2:
            busy(load)
        for mine, theirs in zip(jax.tree.leaves(run(*operands)), jax.tree.leaves(first), strict=True):
            np.testing.assert_array_equal(np.asarray(mine), theirs, err_msg=f"run {repeat}")


def a_layer_choosing_the_fused_step_decodes_on_a_mesh_as_on_one_device(axis: str):
    """A gated delta net choosing the fused step (`generation.chosen`) over two GPUs, its rows (`axis`
    'data') or its heads ('tensor') split two ways (`_fused_token`: each shard's channels its heads'
    queries, keys and values): 9 decode tokens after a 3-token prefill on par (check 1) with the same
    layer's on one device, from the float64 parallel forward, over drawn hidden states and weights."""
    module = GatedDeltaNet(emb_features=64, num_k_heads=2, num_v_heads=4, head_k_dim=32, head_v_dim=32,
                           dtype=jnp.float32)
    # The data axis takes the devices the others leave.
    mesh = MeshSpec(**({} if axis == "data" else {axis: 2})).build(jax.devices()[:2])

    def draw(seed):
        hidden = jax.random.normal(jax.random.key(13 + 2 * seed), (2, 12, 64), jnp.float32)
        return module.init(jax.random.key(14 + 2 * seed), hidden[:, :1]), hidden

    def decoded(variables, hidden):
        return decode_through(module, variables, hidden, prefill=3)[0]

    split_run = jax.jit(named(decoded, delta_decode="fused"))

    @as_given
    def split(variables, hidden):
        with jax.set_mesh(mesh):
            return split_run(variables, jax.device_put(hidden, NamedSharding(mesh, PartitionSpec())))

    assert_as_exact(split, named(decoded, delta_decode="fused"), draw,
                    f"a layer choosing the fused step, decoding split over {axis}", seeds=FEWEST_SEEDS,
                    truth_fn=module.clone(dtype=jnp.float64).apply)


STRESS = "a stress regime a model's draws do not reach"
"""Why a cell keeps its regime on the GPU (Main, 11-11): the template's and the SSD port's stress inputs,
where the GPU's own rounding (TF32, `ex2.approx`) meets a regime a flow's ordinary draws never show, every
one of them. The template's 120 took 1,851 s on an A100 and 1,578 s on a GH200 (the tier at 173b18381);
interpreted, their gradients' declared draws would take about 22 CPU-hours a run, and a long sequence's and
a fading memory's are past a task, so the CPU interprets each form's forward in its ordinary regimes alone
(test_recurrence_template_stability's sharded twins)."""

USERS_PRODUCTS = "the portable math's state products at a user's process's precision (M-JS-96)"
"""Why the portable backward's cells stay on the GPU, every one: in a user's A100 process its state products,
stating no precision, ran in TF32, 1823 times the fp32 recurrence's distance from float64, which the
interpreter, exact, and the suite's process, at 'highest', never show."""

RACES = "a race or an asynchronous order the interpreter's one schedule cannot show"
"""Why a repeat-bitwise cell stays on the GPU: the interpreter runs one schedule (the recurrence lane, 11-11:
07cf6603271's register reuse differed run to run on a GH200 while every CPU check passed)."""

PACKED_PORT = "the FLA port's packed training, which no flow packs"
"""Why a port's packed training stays on the GPU: no model flow packs documents."""

FORKED = "a forked call's walk over its table, which no flow forks"
"""Why a forked call stays on the GPU: its walks take each document's memory up from the call's table and
store the finals later documents take, across launches whose order the interpreter's one schedule fixes
(the recurrence lane, 11-11), and no model flow forks a call."""

NATIVE_ROUNDING = "an fp32 call's products and exponentials at the GPU's own rounding"
"""Why an fp32 call at the default precision stays on the GPU: its products round as each instruction family
rounds an fp32 operand (mma.sync's TF32 'rna', a warpgroup MMA's TF32 read; the run's fast forms, the
owner's "tf32 default") and its exponentials are ex2.approx's, which the interpreter computes exactly (the
recurrence lane, 11-11): a cell each GPU kind runs, its family's own."""


def template_stability_cells() -> list[Cell]:
    """The template finite and on par where chunked forms break, every form, regime and product."""
    finite = template_stability.the_template_stays_finite_and_on_par_where_chunked_forms_break
    return [Cell(f"the template stays finite and on par where chunked forms break: {name} {regime} {product}"
                 f"{'' if forms is None else f' ({forms} forms)'}",
                 functools.partial(finite, name, regime, product, forms), fault=STRESS)
            for name, regime, product, forms in template_stability.CASES]


def portable_cells() -> list[Cell]:
    """The portable backward as a GPU compiles it in a user's process (`USERS`): its products at the
    precision a user's run gives them, JAX's default unless they state one."""
    exact = stability.test_the_chunked_ops_gradient_is_as_exact_as_the_recurrences
    label = "the portable backward as a GPU compiles it is as exact as the recurrence's"
    return [Cell(f"{label}: {name} {regime}", functools.partial(exact, name, regime), fault=USERS_PRODUCTS)
            for name, regime in stability.ACCELERATED]


def ssd_port_cells() -> list[Cell]:
    """The SSD port's forward and gradient under every regime, and the SSD on the template against
    mamba_ssm."""
    return [
        *(Cell(f"the SSD port stays finite and exact where chunked forms break: {regime}",
               functools.partial(stability.the_ssd_port_stays_finite_and_exact_where_chunked_forms_break,
                                 regime), fault=STRESS)
          for regime in stability.REGIMES),
        Cell("the SSD on the template under the fast forms is as exact as mamba_ssm",
             mamba_ssd.the_template_is_as_exact_as_mamba_ssm, native={"launch": "ssd template"}),
        *(Cell(f"the SSD port's gradient is as exact as the recurrence's: {regime}",
               functools.partial(stability.the_ssd_ports_gradient_is_as_exact_as_the_recurrences, regime),
               fault=STRESS)
          for regime in stability.REGIMES),
    ]


def selective_cells() -> list[Cell]:
    """A Mamba layer under the template's selective scan."""
    return [
        *(Cell(f"a Mamba layer is as exact under the template: packed {packed}"
               f"{f', a held state {held}' if held else ''}",
               functools.partial(selective_template.the_layer_is_as_exact_under_the_template, packed, held),
               "sm90", native={"documents": "packed" if packed else "one", "state": str(held)})
          for packed, held in ((False, None), (True, None), (False, "through the template"))),
    ]


def port_training_cells() -> list[Cell]:
    """Each recurrence layer trained on its port's kernels, its step repeating bit for bit, and the ports
    against fla's own."""
    return [
        Cell(f"a gated delta net trains on its port on par with the portable rule ({PORT_KERNELS['fla']})",
             functools.partial(trains_on_its_port, gated_delta_net), fault=PACKED_PORT),
        Cell(f"a Kimi delta attention layer trains on its port on par with the portable rule "
             f"({PORT_KERNELS['kda']})", functools.partial(trains_on_its_port, kimi_delta_attention),
             fault=PACKED_PORT),
        # Kimi delta attention trains on its port alone: the template's fast backward takes no decay a key
        # dimension.
        *(Cell(f"a {make.__name__} training step on {kernel} repeats bit for bit",
               functools.partial(repeats_bit_for_bit, make, kernel), fault=RACES)
          for make, kernels in ((gated_delta_net, ("fla", "mosaic_gpu")), (kimi_delta_attention, ("fla",)))
          for kernel in kernels),
        *(Cell(f"the port is as exact as fla's own: {'kda' if per_key else 'gdn'} {dtypes}",
               functools.partial(fla_gdn.the_port_is_as_exact_as_fla, per_key, dtypes),
               native={"kernel": "kda" if per_key else "gdn", "dtypes": dtypes})
          for per_key in (False, True) for dtypes in fla_gdn.FLA_DTYPES),
    ]


def forked_cells() -> list[Cell]:
    """The template over a forked call."""
    return [Cell(f"the template's forked documents are as exact as each alone: {hooks}",
                 functools.partial(chunked.the_templates_forked_documents_are_as_exact_as_each_alone, hooks),
                 fault=FORKED)
            for hooks in chunked.FORKED_TEMPLATE]


def fp32_template_cells() -> list[Cell]:
    """The template's fp32 forward and training step at the default precision on each GPU's instruction
    family, in a user's process (`USERS`), where the default precision is JAX's own."""
    return [
        # The fp32 forward each instruction family takes at a GPU's widths (an sm80's, an sm90's, an sm100's;
        # jsf-fp32-taken), its training an sm90's alone.
        Cell("the template's fp32 forward at the default precision is as exact as fla's: one sequence",
             functools.partial(forward.the_mosaic_forward_is_as_exact_as_its_reference, "one sequence",
                               jnp.float32), fault=NATIVE_ROUNDING),
        Cell("the template's fp32 training step at the default precision is as exact as fla's: one sequence",
             functools.partial(gradient.the_mosaic_gradient_is_as_exact_as_its_reference, "one sequence",
                               jnp.float32, "kept"), "sm90", fault=NATIVE_ROUNDING),
    ]


def walk_cells() -> list[Cell]:
    """A two-role walk's training on an H100 (`Tiles.walkers`): its two consumers' synchronization and stores,
    which the derivation gives a call whose walks fill more than a wave, a launch no flow's widths reach."""
    walk = gradient.test_a_walk_of_two_roles_is_as_exact_as_its_reference
    return [Cell(f"a walk of two roles is as exact as its reference: {name}",
                 functools.partial(walk, *values), "sm90", fault="two consumer roles' synchronization")
            for name, values in forward.native(gradient.WALK_GRID, gradient.INTERPRETED_WALKS)]


def decode_cells() -> list[Cell]:
    """The decode launches as a server runs them."""
    return [
        Cell("the decode kernel leaves an idle row alone", the_decode_kernel_leaves_an_idle_row_alone,
             native={"launch": "decode step"}),
        Cell("the fused decode step gives the same bits however its launches are timed",
             the_fused_decode_step_gives_the_same_bits_however_its_launches_are_timed, fault=RACES),
        *(Cell(f"a layer choosing the fused step decodes on a mesh as on one device: {axis}",
               when(jax.device_count() >= 2, "needs two GPUs", functools.partial(
                   a_layer_choosing_the_fused_step_decodes_on_a_mesh_as_on_one_device, axis)),
               fault="two GPUs")
          for axis in ("data", "tensor")),
    ]


FAMILIES = {"two-role walk": walk_cells, "template stability": template_stability_cells,
            "portable": portable_cells, "SSD port": ssd_port_cells, "selective scan": selective_cells,
            "forked template": forked_cells, "fp32 template": fp32_template_cells,
            "port training and decode": lambda: [*port_training_cells(), *decode_cells()]}
"""The recurrence cells by family, each family a test of its own: the planner times and packs each, and one
family's failure hides no other's (a single flow of every family ran past a GPU job's 50 minutes). A
layer's port training and its decode run as one flow. The Mosaic mainloop's forward and gradient on a GPU are
the model flows' (tests/test_gpu_flows.py: Kimi's delta attention, Qwen3-Next's gated delta net, MiniMax's
lightning attention), its cases the CPU's, interpreted and through the compile gate (row 5461)."""

USERS = {"portable", "fp32 template"}
"""The families whose claim is what a user's process computes, run in a process computing as a user's does
(`users_process`, tests/conftest.py) and not in the suite's, whose JAX_DEFAULT_MATMUL_PRECISION=highest
(tests/lane_environment.py) gives every product that states no precision IEEE float32: the portable math's
state products, stating none, ran there exact and in a user's A100 process as TF32, 1823 times the fp32
recurrence's distance from float64 (M-JS-96), which no cell saw (the recurrence lane, 11-11)."""


@pytest.mark.parametrize("family", [
    pytest.param(family, marks=pytest.mark.users_process if family in USERS else ()) for family in FAMILIES])
def test_the_recurrences_run_on_this_gpu(family):
    """A family's recurrence cells on this GPU (`gpu_flow.run_cells`) a GPU runs (`gpu_flow.covering`: each
    kept for a GPU-only fault by name, the fewest others covering its native axes, every other the CPU's,
    interpreted and through the compile gate), each one's failure kept and all raised together."""
    run_cells(covering(FAMILIES[family]()))


def test_the_recurrence_matrix_holds_on_this_gpu():
    """The recurrence's check-1 matrix (tests/kernel_matrix.py): the chunked op's training step, its kernel
    named, on par with its form's port's, or the portable math's where no port takes it, at each combination
    of the axes a GPU runs natively (`kernel_matrix.native_cells`), each cell on its own GPU."""
    run_cells(kernel_matrix.seeded_cells("recurrence"))


def test_the_short_conv_matrix_holds_on_this_gpu():
    """The short conv's check-1 matrix: a delta net's conv at its training shapes on the kernels, on par with
    XLA's conv, in each precision on each GPU (`kernel_matrix.native_cells`)."""
    run_cells(kernel_matrix.seeded_cells("short_conv"))
