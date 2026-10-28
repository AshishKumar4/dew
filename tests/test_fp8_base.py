"""A LoRA run's frozen base multiplied in FP8 (`LoRA.fp8_base`).

The sites a record names (`InputQuantization.frozen_fp8`) hold their frozen
kernels packed once, from the base the adapter freezes, as E4M3 rows for the
forward product and for the input gradient (`frozen_fp8_leaves`); each such
Dense computes its output from them inside the adapter's branch, its input's
rows cast by the same FP8 block cast (`dew.interop.codecs`). The product is
held to the float64 product of those exact operands, on one device and split
over fsdp, every site of a model is checked to take it, and a run with an FP8
base resumes and scans as one with a bf16 base does. A whole run split over
fsdp is not held to one device's: a last-bit difference upstream moves an
activation across an E4M3 rounding boundary. Where FP8 pays is a GPU's
measurement (`measured_fp8_sites`), recorded with the generation it was
taken on.
"""


import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from affine_run import Replaying
from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.sharding import NamedSharding, PartitionSpec
from model_support import TINY_DECODER
from process_support import READ_ONLY, fresh, keeping_picks
from reference_error import as_given, assert_as_exact, widened, with_gradients, without_ynn

from dew.batch import FROZEN
from dew.data.dataset import Loader
from dew.interop.codecs import fp8_block_cast, quantize_fp8_blocks
from dew.lora import LoRA, unadapted
from dew.nn.kernels.generation import assumed, device_generation, fp8_tensor_cores
from dew.objectives.lm import LMObjective
from dew.quantization import (
    FROZEN_FP8,
    QUANTIZATION,
    InputQuantization,
    InputQuantizer,
    frozen_fp8_kernel,
    frozen_fp8_leaves,
    frozen_fp8_product,
    measured_fp8_sites,
)
from dew.registry import models, wrapper_layers
from dew.training import Checkpoints, Layout, MeshSpec, Trainer

VOCAB, SEQ, BATCH = 64, 16, 8
FEATURES, MLP = 64, 128
FP8_SITES = ("q_proj", "gate_proj", "up_proj", "down_proj", "lm_head")
"""The FP8 sites, the untargeted vocabulary head among them."""


@pytest.mark.mesh(devices=2)
def test_a_resident_frozen_kernel_packs_in_a_sharded_program():
    """Packing a resident base executes in its sharded JAX program, without gathering the kernel
    to NumPy. Both forward and input-gradient encodings equal the checkpoint codec's bytes."""
    from jax.sharding import Mesh, PartitionSpec as P

    matrix = np.random.default_rng(37).normal(size=(32, 48)).astype(np.float32)
    mesh = Mesh(np.asarray(jax.devices()[:2]), ("model",))
    placed = jax.device_put(matrix, NamedSharding(mesh, P("model", None)))
    actual = jax.jit(frozen_fp8_leaves)(placed)
    hosted = frozen_fp8_leaves(matrix)
    assert all(isinstance(leaf, np.ndarray) for leaf in hosted.values())
    forward, forward_scale = quantize_fp8_blocks(matrix.T, (1, matrix.shape[0]))
    backward, backward_scale = quantize_fp8_blocks(matrix, (1, matrix.shape[1]))
    for key, expected in zip(FROZEN_FP8, (forward, forward_scale, backward, backward_scale), strict=True):
        np.testing.assert_array_equal(np.asarray(actual[key]), expected)
        np.testing.assert_array_equal(hosted[key], expected)


def decoder(**overrides):
    fields = {**TINY_DECODER, "vocab_size": VOCAB, "emb_features": FEATURES, "mlp_features": MLP,
              "dtype": "bfloat16", "tie_embeddings": False}
    return models.build("causal_transformer", **{**fields, **overrides})


def ids():
    return jnp.asarray(np.random.default_rng(0).integers(0, VOCAB, (BATCH, SEQ)), jnp.int32)


def record(variables, generation=None) -> InputQuantization:
    """The FP8 sites of the tiny decoder as a hand-written record, measured
    on this process's own backend, where it multiplies them."""
    kernels = flatten_dict(variables["params"], sep="/")
    return InputQuantization(
        frozen_fp8=tuple(path.removesuffix("/kernel") for path in kernels
                         if path.endswith("/kernel") and path.split("/")[-2] in FP8_SITES),
        measured_on=generation or device_generation())


def sites(adapter) -> tuple[str, ...]:
    """The frozen FP8 sites of the record `adapter`'s model computes under."""
    return adapter.model._dew_wrapped._dew_wrapper().frozen_fp8


def adapters(model=None, generation=None):
    """One adapter over a bf16 base and one over the same base in FP8, its record measured on
    `generation` (this process's own by default)."""
    model = decoder() if model is None else model
    variables = model.init(jax.random.key(0), ids())
    spec = LoRA(rank=4, modules=("q_proj", "v_proj"))
    return (spec.apply(model, variables, key=1),
            spec.apply(model, variables, key=1, fp8_base=record(variables, generation)))


ELSEWHERE = "sm90" if device_generation() == "sm80" else "sm80"
"""A generation other than this process's: an A100's record runs on an H100, anything else's on an A100."""


def calls(model, variables, inputs):
    """Every Dense call of `model`'s forward: its path, the module unbound, its input and its output."""
    seen = {}

    def capture(next_fun, args, kwargs, context):
        output = next_fun(*args, **kwargs)
        if context.method_name == "__call__" and isinstance(context.module, nn.Dense):
            seen["/".join(context.module.path)] = (context.module.clone(parent=None), args[0], output)
        return output

    with nn.intercept_methods(capture):
        model.apply(variables, inputs)
    return seen


PRODUCT_SEEDS = 16
SCANNED_SEEDS = 8
"""The scanned stack's declared draws, sized by `seeds_for` (on armada)."""
"""The frozen product's declared draws (check 1, `assert_as_exact`), sized by `seeds_for`: the fewest for
which the float32 reference compared with itself resolves the factor."""


def product_case(seed=1):
    """An input whose rows span twelve binades, a kernel packed, and an output cotangent:
    `(x, forward, forward_scale, dgrad, dgrad_scale, cotangent)`."""
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.standard_normal((48, 96)) * np.exp2(rng.integers(-6, 6, (48, 1))), jnp.bfloat16)
    kernel = rng.standard_normal((96, 80)).astype(np.float32) * 0.05
    leaves = [jnp.asarray(value) for value in frozen_fp8_leaves(kernel).values()]
    return x, *leaves, jnp.asarray(rng.standard_normal((48, 80)), jnp.float32)


def fp8_products(x, forward, forward_scale, dgrad, dgrad_scale, cotangent):
    """Dew's frozen product and the input gradient it takes under `cotangent` (`with_gradients`)."""
    product = with_gradients(lambda x: frozen_fp8_product(x, forward, forward_scale, dgrad, dgrad_scale))
    return product(x, cotangent)


def dequantized_products(x, forward, forward_scale, dgrad, dgrad_scale, cotangent):
    """The reference arm: the FP8 block cast's rows of `x` and of the cotangent and the packed rows,
    each dequantized, multiplied in float32 by XLA, the input gradient rounded to `x`'s dtype; the
    same products Dew computes from the codes, with the scales applied before the dot rather than
    after it."""
    def product(values, rows, row_scales):
        codes, scales = fp8_block_cast(values, (1, values.shape[1]))
        cast = codes.astype(jnp.float32) * scales
        return jnp.dot(cast, (rows.astype(jnp.float32) * row_scales).T, precision=jax.lax.Precision.HIGHEST,
                       preferred_element_type=jnp.float32)

    return product(x, forward, forward_scale), product(cotangent, dgrad, dgrad_scale).astype(x.dtype)


def exactly(codes, scales) -> np.ndarray:
    """What FP8 rows `codes` scaled a row by `scales` `[rows, 1]` stand for, in float64: each code times
    its row's scale, unrounded."""
    return np.asarray(codes).astype(np.float64) * np.asarray(scales).astype(np.float64)


def exact_products(x, forward, forward_scale, dgrad, dgrad_scale, cotangent):
    """The float64 products of the FP8 block cast's rows and the packed rows, each code times its scale
    in float64, as the codes stand for them."""
    def product(values, rows, row_scales):
        codes, scales = quantize_fp8_blocks(np.asarray(values, np.float32), (1, values.shape[1]))
        return exactly(codes, scales) @ exactly(rows, row_scales).T

    return product(x, forward, forward_scale), product(cotangent, dgrad, dgrad_scale)


def assert_products_exact(products, label):
    """The forward and the input gradient `products` computes (`fp8_products`, placed as a run places
    them) are on par with the dequantized float32 products over the site's draws (check 1), measured from
    the float64 products of the FP8 block cast's rows."""
    assert_as_exact(products, jax.jit(dequantized_products), product_case, label,
                    seeds=PRODUCT_SEEDS, truth_fn=exact_products)


def test_the_frozen_product_multiplies_the_cast_rows_exactly_and_trains_no_kernel():
    """Neither the packed kernel nor its scales take a gradient."""
    x, *leaves, cotangent = product_case()
    _, pullback = jax.vjp(frozen_fp8_product, x, *leaves)
    gradients = pullback(cotangent)
    assert gradients[0].dtype == x.dtype and all(gradient is None or not np.any(gradient)
                                                 for gradient in gradients[1:])
    assert_products_exact(jax.jit(fp8_products), "the frozen FP8 product")


TANGENT_SEEDS = 16
"""The forward-mode product's declared draws (check 1, `assert_as_exact`), sized by `seeds_for` (on
armada): the fewest for which the float32 reference compared with itself resolves the factor."""


def tangent_case(seed=1):
    """`product_case`'s input and packed kernel, and a direction for the input: `(x, forward,
    forward_scale, dgrad, dgrad_scale, direction)`."""
    x, *leaves, _ = product_case(seed)
    direction = jnp.asarray(np.random.default_rng([7, seed]).standard_normal(x.shape), x.dtype)
    return x, *leaves, direction


def tangents(x, forward, forward_scale, dgrad, dgrad_scale, direction):
    """Dew's frozen product in forward mode (`jax.jvp`): its value and its tangent along `direction`."""
    return jax.jvp(lambda x: frozen_fp8_product(x, forward, forward_scale, dgrad, dgrad_scale), (x,),
                   (direction,))


# Both arms of the forward-mode check are Dew programs, compiled alike (docs/design/testing.md).
fp8_tangents = without_ynn(tangents)


@as_given
def default_precision_tangents(*inputs):
    """`fp8_tangents` compiled under JAX's default matmul precision, not the suite's highest
    (`lane_environment`): the products state their own."""
    with jax.default_matmul_precision("default"):
        return fp8_tangents(*inputs)


def decoded_tangents(x, forward, forward_scale, dgrad, dgrad_scale, direction):
    """The reference arm: the dequantized forward product (`dequantized_products`) and the direction
    times the decoded forward rows, each code times its row's scale, multiplied in float32 by XLA."""
    codes, scales = fp8_block_cast(x, (1, x.shape[1]))
    rows = forward.astype(jnp.float32) * forward_scale
    value = jnp.dot(codes.astype(jnp.float32) * scales, rows.T, precision=jax.lax.Precision.HIGHEST,
                    preferred_element_type=jnp.float32)
    return value, jnp.dot(direction.astype(jnp.float32), rows.T, precision=jax.lax.Precision.HIGHEST,
                          preferred_element_type=jnp.float32)


def exact_tangents(x, forward, forward_scale, dgrad, dgrad_scale, direction):
    """The float64 product of the cast rows (`exact_products`) and the direction times the decoded
    forward rows in float64."""
    codes, scales = quantize_fp8_blocks(np.asarray(x, np.float32), (1, x.shape[1]))
    rows = exactly(forward, forward_scale)
    return exactly(codes, scales) @ rows.T, np.asarray(direction, np.float64) @ rows.T


def test_the_frozen_product_in_forward_mode_is_the_decoded_kernels_tangent():
    """Forward mode (`jax.jvp`, MeanFlow's target) takes the product's value as the FP8 product and its
    tangent as the input's direction times the decoded kernel (`frozen_fp8_kernel`), the linear map the
    FP8 products stand for, each on par with the dequantized float32 products by check 1 from float64,
    whatever JAX's default matmul precision; the packed leaves take no tangent."""
    assert_as_exact(default_precision_tangents, without_ynn(decoded_tangents), tangent_case,
                    "the frozen FP8 product in forward mode", seeds=TANGENT_SEEDS, truth_fn=exact_tangents)
    x, *leaves, _ = tangent_case()
    _, along_leaves = jax.jvp(frozen_fp8_product, (x, *leaves),
                              (jnp.zeros_like(x), *(jnp.ones_like(leaf) for leaf in leaves)))
    assert not np.any(np.asarray(along_leaves))


def test_the_frozen_products_backward_keeps_its_packed_leaves_and_not_its_input():
    """Reverse mode reads the input's dtype alone, so the residuals its pullback keeps (the leaves of
    the `jax.vjp` pullback, a `jax.tree_util.Partial`) are the packed leaves and never the `[M, in]`
    input, as a staged pullback, a remat or a scan would hold them."""
    x, *leaves, _ = product_case()
    _, pullback = jax.vjp(lambda x, *leaves: jnp.sum(frozen_fp8_product(x, *leaves)), x, *leaves)
    saved = jax.tree.leaves(pullback)
    assert saved
    kept = [(leaf.shape, leaf.dtype) for leaf in saved]
    assert (x.shape, x.dtype) not in kept, kept
    assert all((held.shape, held.dtype) in kept for held in leaves[2:]), kept


def test_meanflow_trains_an_adapter_over_an_fp8_base():
    """MeanFlow's target is its model's JVP in time (`jax.jvp`), which runs through each frozen FP8 site
    in forward mode, and its loss's gradient runs through them in reverse: Adam steps on an adapter over
    a DiT whose MLP projections are FP8 sites keep the loss finite and move every factor the adapter
    trains, and the packed base stays out of what trains."""
    from dew.diffusion import presets
    from dew.inputs import CharTable, Condition, Field, InputSpec
    from dew.nn.backbones import SimpleDiT
    from dew.objectives.base import Step
    from dew.objectives.diffusion.few_step import MeanFlowObjective

    model = SimpleDiT(patch_size=2, emb_features=16, num_layers=1, num_heads=2, mlp_ratio=1, interval=True)
    inputs = InputSpec(Field("image", (4, 4, 3)),
                       {"textcontext": Condition(CharTable.from_pretrained("char_table"))})
    plain = MeanFlowObjective(model, presets.MeanFlow()(), inputs, ema_decay=None)
    variables = plain.model_variables(plain.init(jax.random.key(0)))
    # adaLN-Zero's gates and output start at zero, where no gradient reaches the MLP; drawn weights let it.
    leaves, structure = jax.tree.flatten(variables["params"])
    keys = jax.random.split(jax.random.key(4), len(leaves))
    variables = {**variables, "params": jax.tree.unflatten(structure, [
        0.3 * jax.random.normal(key, leaf.shape, leaf.dtype)
        for key, leaf in zip(keys, leaves, strict=True)])}
    kernels = flatten_dict(variables["params"], sep="/")
    named = tuple(path.removesuffix("/kernel") for path in kernels
                  if path.endswith("/kernel") and path.split("/")[-2] in ("up_proj", "down_proj"))
    assert named
    fp8_base = InputQuantization(frozen_fp8=named, measured_on=device_generation())
    adapter = LoRA(rank=2, modules=("up_proj",)).apply(model, variables, key=1, fp8_base=fp8_base)
    assert set(sites(adapter)) == set(named)
    # The objective's tree is the adapter's beside the condition encoders' own.
    objective = MeanFlowObjective(adapter.model, presets.MeanFlow()(), inputs, ema_decay=None,
                                  variables={**plain.init(jax.random.key(0)), **adapter.variables})
    tree = objective.init(jax.random.key(0))
    batch = {"image": np.asarray(jax.random.randint(jax.random.key(1), (4, 4, 4, 3), 0, 256), np.uint8),
             **inputs.tokenize(["a", "b", "c", "d"])}

    def loss(params, step):
        return objective.scalar_loss({**tree, "params": params}, batch,
                                     Step(step=jnp.asarray(step), key=jax.random.key(2), ema=None))[0]

    optimizer = optax.adam(1e-2)
    params, state = tree["params"], optimizer.init(tree["params"])
    for step in range(3):
        value, gradients = jax.value_and_grad(loss)(params, step)
        assert np.isfinite(float(value))
        updates, state = optimizer.update(gradients, state, params)
        params = optax.apply_updates(params, updates)
    assert QUANTIZATION in tree and all(path.endswith(("lora_A", "lora_B")) for path in
                                        flatten_dict(tree["params"], sep="/"))
    assert all(not np.array_equal(np.asarray(after), np.asarray(before)) for after, before in
               zip(jax.tree.leaves(params), jax.tree.leaves(tree["params"]), strict=True))


@pytest.mark.mesh(devices=2)
def test_split_over_fsdp_the_frozen_product_is_one_devices():
    """The input's rows and the packed kernel's split over fsdp, as a run
    places them, the product and the input gradient are the same exact
    products: a row's cast reads the row alone, wherever it lies."""
    mesh = jax.make_mesh((2,), ("fsdp",), (jax.sharding.AxisType.Auto,), devices=jax.devices()[:2])
    split = NamedSharding(mesh, PartitionSpec("fsdp", None))
    products = jax.jit(fp8_products)

    @as_given
    def placed(*arrays):
        # Placed before the jit, as a run's arrays arrive: a device_put inside one constrains nothing.
        arrays = jax.device_put(arrays, split)
        assert all(array.sharding == split for array in arrays)
        return products(*arrays)

    assert_products_exact(placed, "the frozen FP8 product split over fsdp")


@pytest.mark.parametrize("measured_on", [None, ELSEWHERE], ids=["here", "elsewhere"])
def test_every_named_site_computes_from_its_packed_kernel_inside_the_branch(measured_on):
    """A fresh adapter's branch adds zero (B starts at 0), so each named
    site's output is the frozen product of its own input, the adapted ones
    included, and every other Dense computes as the bf16 base's does. Where
    the record's sites were measured is provenance: one measured on another
    generation computes the same products here."""
    plain, fp8 = adapters(generation=measured_on)
    seen, base = calls(fp8.model, fp8.variables, ids()), calls(plain.model, plain.variables, ids())
    named = set(sites(fp8))
    assert named and named <= set(seen)
    packed = flatten_dict(fp8.variables[QUANTIZATION], sep="/")
    frozen = fp8.variables[FROZEN]
    for path, (module, x, output) in seen.items():
        if path in named:
            leaves = [packed[f"{path}/{leaf}"] for leaf in FROZEN_FP8]
            expected = frozen_fp8_product(x.reshape(-1, x.shape[-1]), *leaves).astype(output.dtype)
        else:
            own = frozen
            for name in path.split("/"):
                own = own[name]
            expected = module.apply({"params": own}, x)
        np.testing.assert_array_equal(np.asarray(output), np.asarray(expected).reshape(output.shape))
    assert set(base) == set(seen)


def fit(adapter, steps, directory=None):
    objective = LMObjective(adapter.model, SEQ, variables=adapter.variables)
    batches = np.random.default_rng(2).integers(0, VOCAB, (8, BATCH, SEQ + 1)).astype(np.int32)

    mesh = MeshSpec()
    trainer = Trainer(objective, optax.adam(1e-2), key=jax.random.key(0), mesh=mesh,
                      layout=Layout(min_shard=1, tolerance=1.0),
                      checkpoints=None if directory is None else Checkpoints(str(directory)))
    trainer.device_mesh = mesh.build(jax.devices()[:1])
    def windows(position):
        return {"text": batches[position % len(batches)]}

    return trainer.fit(Loader(lambda partition: Replaying(windows), None, None, BATCH), steps=steps,
                       checkpoint_every=None if directory is None else 2)


def test_lora_packs_the_frozen_base_once_in_place_of_its_kernels_and_training_moves_neither():
    """The packed leaves are the frozen kernels' (`frozen_fp8_leaves`) and
    stand in their place, so the state holds no FP8 site's kernel; a merge
    reads the kernel the products ran on (`frozen_fp8_kernel`); and a run
    trains the factors alone: the frozen tree and the packing come out as
    they went in."""
    plain, fp8 = adapters()
    base = flatten_dict(plain.variables[FROZEN], sep="/")
    frozen = flatten_dict(fp8.variables[FROZEN], sep="/")
    packed = flatten_dict(fp8.variables[QUANTIZATION], sep="/")
    merged = flatten_dict(fp8.merge(fp8.variables)["params"], sep="/")
    for site in sites(fp8):
        assert f"{site}/kernel" not in frozen
        leaves = frozen_fp8_leaves(base[f"{site}/kernel"])
        for leaf, value in leaves.items():
            np.testing.assert_array_equal(np.asarray(packed[f"{site}/{leaf}"]).view(np.uint8),
                                          np.asarray(value).view(np.uint8))
        np.testing.assert_array_equal(np.asarray(merged[f"{site}/kernel"]), frozen_fp8_kernel(leaves))
    state = fit(fp8, 3)
    for tree, before in ((state.variables[FROZEN], frozen), (state.variables[QUANTIZATION], packed)):
        assert set(flatten_dict(tree, sep="/")) == set(before)
        for path, value in flatten_dict(tree, sep="/").items():
            np.testing.assert_array_equal(np.asarray(value).view(np.uint8),
                                          np.asarray(before[path]).view(np.uint8))


def test_a_merged_fp8_base_computes_the_kernels_its_training_moved():
    """A merge decodes each FP8 site's kernel and adds its trained delta, and
    the codes go with the kernel they stood for: the model without its
    adapter computes the merged tree as the plain decoder computes its
    kernels, which the trained deltas moved off the decoded base. The head is
    tied, so no packed vocabulary head plays a part."""
    model = decoder(tie_embeddings=True)
    _, fp8 = adapters(model)
    state = fit(fp8, 3)
    merged = fp8.merge(state.variables)
    expected = np.asarray(model.apply({"params": merged["params"]}, ids()), np.float32)
    np.testing.assert_array_equal(np.asarray(unadapted(fp8.model).apply(merged, ids()), np.float32), expected)
    base = fp8.merge(fp8.variables)["params"]
    assert not np.array_equal(expected, np.asarray(model.apply({"params": base}, ids()), np.float32))


def test_a_resumed_fp8_base_run_trains_what_an_uninterrupted_one_does(tmp_path):
    _, fp8 = adapters()
    whole = fit(fp8, 4)
    fit(fp8, 2, tmp_path)
    resumed = fit(fp8, 4, tmp_path)
    for got, want in zip(jax.tree.leaves(resumed.variables), jax.tree.leaves(whole.variables), strict=True):
        np.testing.assert_array_equal(np.asarray(got).view(np.uint8), np.asarray(want).view(np.uint8))


def test_the_scanned_stack_reads_each_layers_own_packed_kernel():
    """The scanned stack computes the looped one's output from the same
    variables, both jitted with the variables as arguments, as a step passes
    its state: closed over, the packed codes fold into a float32 GEMM where a
    scan's traced ones take the FP8 one, which rounds its sums otherwise. The
    two on par over drawn tokens (check 1), from the plain decoder in float64
    on the kernels the codes stand for (the adapter's deltas start at zero)."""
    _, fp8 = adapters()
    _, scanned = adapters(decoder(scan_layers=True))
    exact = jax.jit(decoder().clone(dtype=jnp.float64, attention_impl="reference").apply)
    # The merged tree's layout, each packed site's kernel the one its forward codes stand for in float64
    # (`exactly`), not the float32 decode `merge` stores.
    base = flatten_dict(fp8.merge(fp8.variables)["params"])
    packed = flatten_dict(fp8.variables[QUANTIZATION])
    for path, codes in packed.items():
        if path[-1] == FROZEN_FP8[0]:
            base[(*path[:-1], "kernel")] = exactly(codes, packed[(*path[:-1], FROZEN_FP8[1])]).T
    base = unflatten_dict(base)

    def draw(seed):
        return (jnp.asarray(np.random.default_rng([0xF8, seed]).integers(0, VOCAB, (BATCH, SEQ)), jnp.int32),)

    def truth(tokens):
        with jax.enable_x64(new_val=True):
            return exact({"params": widened(base)}, tokens)

    assert_as_exact(lambda tokens: jax.jit(scanned.model.apply)(fp8.variables, tokens),
                    lambda tokens: jax.jit(fp8.model.apply)(fp8.variables, tokens), draw, "scanned FP8 stack",
                    seeds=SCANNED_SEEDS, truth_fn=truth)


def test_a_checkpoints_input_quantization_and_an_fp8_base_share_one_wrapper():
    """A model computing under its checkpoint's input quantization takes the
    FP8 base into the same record, so one interceptor serves both: the site
    rounds its input first, then multiplies in FP8."""
    model = decoder()
    variables = model.init(jax.random.key(0), ids())
    site = "layers_0/self_attn/q_proj"
    loaded = InputQuantization(sites=((site, "fp8"),)).apply(model)
    variables = {**variables, QUANTIZATION: {"layers_0": {"self_attn": {"q_proj": {
        "fp8_input_scale": jnp.asarray(0.01, jnp.float32)}}}}}
    # The record's fp8 input scale is ModelOpt's amax / 448.
    adapter = LoRA(rank=4, modules=("v_proj",)).apply(loaded, variables, key=1, fp8_base=record(variables))
    _, layers = wrapper_layers(type(adapter.model))
    wrappers = [layer._dew_wrapper() for layer in layers]
    joined = [wrapper for wrapper in wrappers if isinstance(wrapper, InputQuantization)]
    assert len(joined) == 1 and joined[0].sites == ((site, "fp8"),) and site in joined[0].frozen_fp8
    _, x, output = calls(adapter.model, adapter.variables, ids())[site]
    rounded = InputQuantizer.fp8(0.01).qdq(x)
    leaves = [flatten_dict(adapter.variables[QUANTIZATION], sep="/")[f"{site}/{leaf}"] for leaf in FROZEN_FP8]
    expected = frozen_fp8_product(rounded.reshape(-1, x.shape[-1]), *leaves).astype(output.dtype)
    np.testing.assert_array_equal(np.asarray(output), np.asarray(expected.reshape(output.shape)))


FP8_CORES = fp8_tensor_cores(device_generation())


@pytest.mark.skipif(FP8_CORES, reason="this GPU has FP8 tensor cores")
def test_a_device_without_fp8_tensor_cores_measures_no_site():
    variables = decoder().init(jax.random.key(0), ids())
    with pytest.raises(ValueError):
        measured_fp8_sites(variables, record(variables).frozen_fp8, 8192)


def test_a_process_sharing_its_gpu_measures_no_site(monkeypatch):
    """A process beside other work on its GPU (gpu-run -n 0.5 asks XLA to
    read its picks and write none) would time FP8 against bf16 unfairly, so
    it is refused before measuring, on a GPU with FP8 tensor cores too: an
    H100 has them, so the read-only autotune cache is what refuses it. The
    control is a process that would measure: neither read-only nor under
    the cuda lane's deterministic ops, which keep no picks at all."""
    from dew import cache

    variables = decoder().init(jax.random.key(0), ids())
    flags = " ".join(flag for flag in keeping_picks().split()
                     if not flag.startswith(READ_ONLY.split("=")[0]))
    monkeypatch.setenv("XLA_FLAGS", flags)
    assert cache.measures_picks()
    monkeypatch.setenv("XLA_FLAGS", f"{flags} {READ_ONLY}")
    assert not cache.measures_picks()
    with assumed("sm90"), pytest.raises(ValueError):
        measured_fp8_sites(variables, record(variables).frozen_fp8, 8192)


QUALIFY = """
import json
import jax
import jax.numpy as jnp
import numpy as np
from dew import cache
from dew.nn.kernels.generation import device_generation
from dew.quantization import frozen_fp8_leaves, frozen_fp8_product, measured_fp8_sites
from dew.telemetry.instrumentation import hlo_instructions

if not cache.measures_picks():
    print(json.dumps(None))
    raise SystemExit
variables = {"params": {"wide": {"kernel": jnp.zeros((4096, 12288), jnp.float32)},
                        "narrow": {"kernel": jnp.zeros((1024, 1024), jnp.float32)}}}
measured = measured_fp8_sites(variables, ("wide", "narrow"), 8192)
rng = np.random.default_rng(3)
leaves = [jnp.asarray(value) for value in
          frozen_fp8_leaves(rng.standard_normal((4096, 12288)).astype(np.float32) * 0.02).values()]

def step(x, cotangent, *leaves):
    output, pull = jax.vjp(lambda x: frozen_fp8_product(x, *leaves), x)
    return output, pull(cotangent.astype(output.dtype))[0]

x = jnp.asarray(rng.standard_normal((8192, 4096)), jnp.bfloat16)
cotangent = jnp.asarray(rng.standard_normal((8192, 12288)), jnp.bfloat16)
program = hlo_instructions(jax.jit(step).lower(x, cotangent, *leaves).compile().as_text())
gemms = sum(instruction.target == "__cublas$lt$matmul$f8" for instruction in program.values())
print(json.dumps({"sites": list(measured.frozen_fp8), "measured_on": measured.measured_on,
                  "generation": device_generation(), "fp8_gemms": gemms}))
"""
"""The qualification, run by a process of its own (`fresh`): the suite's deterministic ops keep its
own process from measuring (`dew.cache.keeps_picks`)."""


@pytest.mark.gpu(reason="FP8 tensor cores and cuBLASLt's FP8 GEMM")
@pytest.mark.skipif(not FP8_CORES, reason="this GPU has no FP8 tensor cores")
def test_a_frozen_sites_products_take_no_fp8_gemm_whose_sum_is_narrower(tmp_path):
    """On a GPU with FP8 tensor cores the frozen product's forward and input gradient compile to no
    cuBLASLt FP8 GEMM, whose running sum on Hopper is narrower than fp32's (`_fp8_rows`), even at the
    wide projection, and the measurement records the sites it took on this generation. The packed
    leaves and the gradient are the step's arguments, as a train state's are. It runs in a process
    without the suite's deterministic ops, as a run that measures does; a GPU shared with other work
    (`READ_ONLY`) measures nothing, fairly, and skips."""
    found = fresh(QUALIFY, cache_dir=tmp_path, flags=keeping_picks())
    if found is None:
        pytest.skip("this GPU is shared with other work (gpu-run -n 0.5), where no process measures")
    assert set(found["sites"]) <= {"wide", "narrow"} and found["measured_on"] == found["generation"]
    assert found["fp8_gemms"] == 0
