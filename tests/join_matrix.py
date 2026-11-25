"""The joined projections' check-1 matrix (tests/kernel_matrix.py): a layer's training step with two of its
projections of one input run as one GEMM on par with the same layer's projections apart.

Each cell runs a layer as a model's block does, the forward and its reverse mode in one program: a gated MLP,
whose gate and up projections the join `gate_up` runs as one, a gated delta net, whose b and a projections the
join `delta_gates` does (`dew.nn.joined.JOINS`, one mechanism, one selector each), or an attention layer,
whose query, key and value projections of unequal widths `qkv` runs side by side (`dew.nn.joined.BESIDE`: one
`gemm` of their columns, its forward one launch or each product its own as the grouped selector takes it, its
input gradient one chained launch, the template named). The
output and the gradients of its input and every parameter (`reference_error.with_gradients`, a drawn
cotangent) are held to the layer with the join named apart (`generation.chosen`, check 1) from the layer in
float64 with them apart, as tests/test_joined.py holds a join.

The kernels under test are the joins', named `joined` where the cells run them (the selector's own pick is the
projections apart where the join measured no faster on the generation, `dew.nn.joined.JOINED_FASTER`): the
cells' `CHOICES` are `gate_up`, `delta_gates` and `qkv`, a cell running its own. Every other choice a layer
reports (the products' `grouped_matmul`, the gated product's `swiglu`, the delta net's `rms_norm`,
`short_conv` and `gated_delta_rule`, the attention layer's `attention`) runs at XLA, and the fused MLP's
`gated_mlp` apart, each named in the cell (`UNFUSED`), so the layer's other kernels leave the join's check as
it is.

What the route serves is read from the join's own selector: the layer traced for its shapes as the case's
generation with the join named, the route its choice reports for the layer's projections
(`joined.join_case`: plain Dense layers of one width, bias, dtype and precision, holding their own kernels)
the `joined` kernel, not the projections apart a failed requirement hands them to. An axis a join's layer has
no use of (a delta net has no gated product, bias or gate multiplier, an attention layer no gated product or
gate multiplier) takes its one value that says so, `none` or `plain`, and no other.
"""

import dataclasses
import functools

import generation_support
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gpu_flow import cell
from jax.sharding import Mesh
from kernel_matrix import AXES
from reference_error import with_gradients
from test_joined import assert_joined_as_exact, drawing, row_joins

from dew.nn.kernels import generation, selection, tuning
from dew.nn.linear import GatedDeltaNet
from dew.nn.mixer_base import MixerContext
from dew.nn.mixers.attention import CausalSelfAttention
from dew.nn.mlp import GatedMLP

ROWS, FEATURES, HIDDEN = (2, 128), 256, 384
"""The layer's input `[batch, tokens]` rows, the features it reads and writes, a gated MLP's hidden width."""
HEADS, HEAD_WIDTH = (2, 4), 64
"""A gated delta net's key and value heads, and the width of each head's key and value; an attention
layer's key heads and query heads (its query, key and value projections 256, 128 and 128 wide)."""
SEEDS = 8
GATE_MULTIPLIER = 0.75
"""A scaled gate's multiplier (Falcon-H1's `mlp_multipliers`)."""
CHOICES = frozenset({"gate_up", "delta_gates", "qkv"})
"""The `generation.CHOICES` entries the cells run, a cell its case's join: each cell asserts that the routes
it records run that join `joined` and that every choice outside `CHOICES` runs unfused (`UNFUSED`), which
together reach them all."""
KERNELS = {choice: frozenset({"joined"}) for choice in CHOICES}
"""The kernels under test of each choice the cells run: the join's projections as one."""
UNFUSED = {**dict.fromkeys(("grouped_matmul", "swiglu", "rms_norm", "short_conv", "gated_delta_rule",
                            "attention"), "xla"), "gated_mlp": "apart"}
"""The layers' other choices, each named unfused in every cell: XLA's, and the fused MLP's projections
apart."""


def _unfused(join: str) -> dict[str, str]:
    """`join`'s cells' other choices (`UNFUSED`): the side-by-side join's products the GEMM template's
    (`grouped_matmul` named `mosaic`), so its cells run the side-by-side forward and the chained input
    gradient natively where XLA's products would stand for them (head review of 4d6c7a69c)."""
    return {**UNFUSED, "grouped_matmul": "mosaic"} if join == "qkv" else UNFUSED


def _applies(case: dict[str, str]) -> bool:
    """Whether `case` names a layer its join has: a gated MLP's activation and options, an attention layer's
    bias or none (`none` activation, a plain gate), or a delta net's none (`none` activation, no bias, a plain
    gate)."""
    if case["join"] == "gate_up":
        return case["activation"] != "none"
    if case["join"] == "qkv":
        return (case["activation"], case["gate"]) == ("none", "plain")
    return (case["activation"], case["bias"], case["gate"]) == ("none", "none", "plain")


def _layer(case: dict[str, str]) -> GatedMLP | GatedDeltaNet | CausalSelfAttention:
    """`case`'s layer: its projections separate parameters, as its checkpoints store them."""
    dtype = jnp.dtype(case["dtype"])
    if case["join"] == "qkv":
        return CausalSelfAttention(MixerContext(
            emb_features=FEATURES, num_heads=HEADS[1], num_kv_heads=HEADS[0], head_dim=HEAD_WIDTH,
            max_seq_len=ROWS[1], attention_bias=case["bias"] == "bias", dtype=dtype))
    if case["join"] == "delta_gates":
        return GatedDeltaNet(emb_features=FEATURES, num_k_heads=HEADS[0], num_v_heads=HEADS[1],
                             head_k_dim=HEAD_WIDTH, head_v_dim=HEAD_WIDTH, dtype=dtype)
    return GatedMLP(hidden_features=HIDDEN, out_features=FEATURES, use_bias=case["bias"] == "bias",
                    activation=case["activation"],
                    gate_multiplier=GATE_MULTIPLIER if case["gate"] == "scaled" else 1.0, dtype=dtype)


def _wide(layer: GatedMLP | GatedDeltaNet | CausalSelfAttention):
    """`layer` computing in float64, the truth's: an attention layer's dtype its context's."""
    if isinstance(layer, CausalSelfAttention):
        return layer.clone(ctx=dataclasses.replace(layer.ctx, dtype=jnp.float64))
    return layer.clone(dtype=jnp.float64)


def _input(case: dict[str, str]):
    """`case`'s input, abstract: `ROWS` rows of `FEATURES` in its dtype."""
    return jax.ShapeDtypeStruct((*ROWS, FEATURES), jnp.dtype(case["dtype"]))


def supported(case: dict[str, str]) -> bool:
    """Whether the route serves `case` (tests/kernel_matrix.py): its layer, which its join has (`_applies`),
    traced for its shapes as its generation within an abstract mesh of one of its devices, the join named and
    the layer's other choices unfused (`_unfused`), reports the `joined` kernel for its join, no kernel it
    names refusing it."""
    if not _applies(case):
        return False
    layer = _layer(case)
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), case["generation"])
    params = jax.eval_shape(layer.init, jax.random.key(0), _input(case))
    try:
        with (generation.assumed(case["generation"]), jax.sharding.use_abstract_mesh(mesh),
              generation.chosen(**_unfused(case["join"]), **{case["join"]: "joined"}),
              generation.routes() as routes):
            jax.eval_shape(layer.apply, params, _input(case))
    except selection.KernelRefused:
        # A kernel the case names refusing it (the side-by-side join's template, `_unfused`, on a GPU or
        # dtype it takes no product of): the route does not serve it.
        return False
    return [route.kernel for route in routes if route.choice == case["join"]] == ["joined"]


CASES = (
    ('sm90', 'gate_up', 'swiglu', 'none', 'plain', 'bfloat16'),
    ('sm80', 'gate_up', 'geglu', 'bias', 'scaled', 'float16'),
    ('sm80', 'delta_gates', 'none', 'none', 'plain', 'float32'),
    ('sm90', 'qkv', 'none', 'bias', 'plain', 'float16'),
    ('sm90', 'gate_up', 'swiglu', 'bias', 'scaled', 'float32'),
    ('sm90', 'gate_up', 'geglu', 'none', 'scaled', 'bfloat16'),
    ('sm90', 'delta_gates', 'none', 'none', 'plain', 'bfloat16'),
    ('sm80', 'gate_up', 'swiglu', 'none', 'plain', 'float16'),
    ('sm80', 'gate_up', 'geglu', 'bias', 'plain', 'bfloat16'),
    ('sm90', 'qkv', 'none', 'none', 'plain', 'bfloat16'),
    ('sm90', 'gate_up', 'geglu', 'none', 'plain', 'float32'),
    ('sm90', 'delta_gates', 'none', 'none', 'plain', 'float16'),
)
"""The cells' cases, each its axis values in `AXES`' order: every pair of axis values the route serves
(`kernel_matrix.covering("join")`, regenerated where the axes or the route's support change)."""


@cell("join", covers=lambda case: dict(zip(AXES["join"], case, strict=True)))
@pytest.mark.parametrize("case", [pytest.param(case, marks=generation_support.pinned(case[0]))
    for case in CASES], ids="-".join)
def a_joined_layers_training_step_is_on_par(case: tuple[str, ...]) -> None:
    """The layer's training step of `case` on its GPU with its two projections joined on par with the
    projections apart (check 1): the output and the gradients of the input and every parameter, from the layer
    with them apart in float64. A gated MLP's biases are drawn too, which init leaves at zero. The routes it
    records run its join `joined` and every other choice unfused (`CHOICES`, `UNFUSED`); a joined attention
    layer's step binds the side-by-side forward and the chained input gradient, apart neither."""
    combination = dict(zip(AXES["join"], case, strict=True))
    layer = _layer(combination)
    dtype = jnp.dtype(combination["dtype"])

    def draw(seed):
        # The layer's parameters drawn with each seed's inputs, the same each draw: a GPU tier's prepare,
        # whose device computes nothing, reaches the site with nothing computed (row 10990).
        # Initialized by XLA's products: a draw runs on the host's CPU and its own kernels
        # (`arm_arithmetic.ahead_on_the_host`), where the template the cell names refuses the call
        # (row 12378); its routes its own, none of the step's the cell reads (row 12868: a GH200's qkv
        # cells read the init's XLA products as the step's).
        with generation.chosen(grouped_matmul="xla"), generation.routes(alone=True):
            params = layer.init(jax.random.key(4), jnp.zeros((*ROWS, FEATURES), dtype))
        if combination["join"] in ("gate_up", "qkv"):
            params = jax.tree.map(lambda leaf: leaf + 0.1 * jax.random.normal(jax.random.key(6), leaf.shape),
                                  params)
        return drawing(params, (*ROWS, FEATURES), dtype)(seed)

    unfused = _unfused(combination["join"])
    with generation.chosen(**unfused), generation.routes() as routes:
        assert_joined_as_exact(combination["join"], with_gradients(layer.apply), draw,
                               with_gradients(_wide(layer).apply), "-".join(case), seeds=SEEDS)
    recorded = {(found.choice, found.kernel) for found in routes}
    unnamed = sorted(pair for pair in recorded
                     if pair[0] not in CHOICES and pair[1] != unfused.get(pair[0], "xla"))
    assert (combination["join"], "joined") in recorded and not unnamed, (sorted(recorded), unnamed)
    if combination["join"] == "qkv":
        # Joined, its step binds the side-by-side forward and the chained input gradient; apart, neither
        # (head review of 0099a4701 F4: the cell checked the route alone).
        operands = jax.eval_shape(draw, 0)
        for joins, forms in ((True, {"beside", "chained"}), (False, set())):
            with generation.chosen(**unfused), row_joins("qkv", joins=joins), tuning.seen() as launches:
                jax.eval_shape(with_gradients(layer.apply), *operands)
            bound = {form for _, each in launches.taken for form in ("beside", "chained")
                     if each.arguments.get(form)}
            assert bound == forms, (joins, bound)


def per_shard(choice: str, kernel: str) -> dict:
    """`kernel`'s per-shard cells (`sharded.Shard`, tests/test_per_shard.py): `choice`'s layer of its first
    16-bit case (`CASES`) over rows a two-device data axis splits, its parameters whole, its two projections
    joined into one product the GEMM template launches (`grouped_matmul` named `mosaic`, interpreted on the
    CPU's devices) per device on its own rows (`joined.py`'s map), its other kernels unfused (`UNFUSED`): the
    output and every gradient on par with one device's from the layer in float64. The fsdp placement of the
    joined product's weights is the GEMM's cells' (`gemm_matrix.per_shard`, joined cases)."""
    from jax.sharding import PartitionSpec
    from sharded import Shard, layer_scope


    del kernel
    # A 16-bit case: the template multiplies no fp32 product above the default precision (the lane's).
    narrow = next(case for case in CASES if case[1] == choice and case[-1] != "float32")
    case = dict(zip(AXES["join"], narrow, strict=True))
    layer = _layer(case)
    dtype = jnp.dtype(case["dtype"])

    @functools.cache
    def params():
        # On the cell's first draw, not where the family states its cells: every per-shard task's
        # collection reads that statement, which initialized each join's layer there (rows 11692, 11695).
        return layer.init(jax.random.key(4), jnp.zeros((*ROWS, FEATURES), dtype))


    def truth(*operands):
        # The layer in float64 through XLA's products, which a float64 product runs on, its projections apart:
        # joined, a named grouped product other than the template's is refused (`joined.BESIDE_SUPPORTS`).
        with generation.chosen(grouped_matmul="xla", **{case["join"]: "apart"}):
            return with_gradients(_wide(layer).apply)(*operands)

    named = {**UNFUSED, "grouped_matmul": "mosaic"}
    return {"data": Shard(with_gradients(layer.apply),
                          lambda seed: drawing(params(), (*ROWS, FEATURES), dtype)(seed),
                          lambda mesh: [PartitionSpec(), PartitionSpec("data"), PartitionSpec("data")], truth,
                          SEEDS, "data", context=lambda: layer_scope(**named))}
