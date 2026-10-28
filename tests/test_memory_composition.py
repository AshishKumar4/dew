"""The in-context memory (`dew.nn.memory`, ledger 1645) composed: as a decoder's layer kind, scanned,
rematted, decoded, under Muon, refused bidirectional, its remat marks, and recorded; its heads' networks
under the model's wrappers, an adapter on every head or one, a checkpoint's input quantization, a frozen
FP8 base, Qwix's quantized training and serving; its decode cache across the routes a call's wrappers
choose; its rows split over a mesh's data axis; and a memory held under a field named `network`."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn, traverse_util
from memory_support import (
    HEADS,
    LENGTH,
    MEMORY_SEEDS,
    ROWS,
    STEP,
    WIDTH,
    applied,
    decoded,
    drawn,
    head_of,
    inner_sites,
    layer,
    written,
)
from reference_error import as_given, assert_as_exact

from dew.batch import coordinates
from dew.nn.inputs import AttentionMetadata, MixerCall
from dew.nn.memory import CARRY, Gate, InContextMemory, MLPMemory, Objective
from dew.nn.mlp import GatedMLP

SEEDS = {"memory decoder gradient": 512, "memory decoder decode": 16, "adapted memory": 8, "rows split": 256,
         "rows split cms": 512, "served memory moving": 8, "served memory projections": 8,
         "served memory convolved": 8}
"""The sites' declared seeds where `seeds_for` on armada, each reference compared with itself, sizes them
apart from `MEMORY_SEEDS`: the memory decoder's gradients, every parameter's leaf, and the
continuum-memory level's rows split over a data axis at 512; the other memories' rows split at 256; the
decoder's last logits after a cached decode at 16; an adapted memory decoder at 8; each served memory
(`SERVED`) at 8, paired with quantized training (`seeds_for(..., dew_fn=)`)."""


def decoder(memory: InContextMemory | None = None, **fields):
    """A tiny decoder of two memory layers, Titans-style where `memory` is None, which a scanned stack
    scans as one group, then two attention layers."""
    from model_support import TINY_DECODER

    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.backbones.layer_plan import LayerKind
    from dew.nn.mixers.memory import InContextMemoryMixer

    if memory is None:
        memory = InContextMemory(network=MLPMemory(hidden=(16,)), heads=2, head_features=8, chunk=4,
                                 normalize=True, learning_rate=Gate(scale=STEP), momentum=Gate(),
                                 retention=Gate(bias=-2.0))
    return CausalTransformer(**{"vocab_size": 37, **TINY_DECODER, "num_layers": 4, "dtype": jnp.float32,
                                "layer_types": ("memory", "memory", "attention", "attention"),
                                "kinds": {"memory": LayerKind(mixer=InContextMemoryMixer(memory))}, **fields})


def test_a_memory_decoder_trains_scans_remats_and_decodes():
    """A decoder whose first two layers are memories, which its scanned stack scans as one group, from
    the plain loop's tree: the scanned stack's logits and gradient, and both under a remat policy that
    keeps each memory's chunk carry (`CARRY`), are the plain loop's, on par against float64 (check 1),
    the gradient reaching every memory's initial network; and the scanned stack's cached decode, token
    by token after a prefill, gives the full pass's last logits (check 1)."""
    from model_support import model_draw

    from dew.nn.backbones.decoder_block import RematPolicy

    plain, scanned = decoder(), decoder(scan_layers=True)
    rematted = decoder(scan_layers=True, remat=RematPolicy(save=(CARRY,)))
    wide = decoder(dtype=jnp.float64)
    draw_model = model_draw(plain, (ROWS, LENGTH), 37)

    def trained(model):
        """The model's logits and its parameters' gradient under a cotangent."""
        @jax.jit
        def run(variables, ids, cotangent):
            logits, pull = jax.vjp(lambda held: model.apply(held, ids), variables)
            return logits, pull(cotangent)[0]["params"]
        return run

    def draw_with_cotangent(seed: int):
        cotangent = np.random.default_rng([1648, seed]).standard_normal((ROWS, LENGTH, 37)).astype(np.float32)
        return *draw_model(seed), cotangent

    for name, model in (("scanned", scanned), ("rematted", rematted)):
        assert_as_exact(trained(model), trained(plain), draw_with_cotangent,
                        f"{name} memory decoder gradient", seeds=SEEDS["memory decoder gradient"],
                        truth_fn=trained(wide))
    gradient = trained(rematted)(*draw_with_cotangent(0))[1]
    for index in (0, 1):
        layer = gradient[f"layers_{index}"]["self_attn"]
        assert all(np.abs(np.asarray(leaf)).max() > 0 for name in layer if head_of((name,)) is not None
                   for leaf in jax.tree.leaves(layer[name]))

    @jax.jit
    def decoded(variables, ids):
        cache = scanned.apply(variables, ids.shape[0], method=scanned.init_cache, mutable=["cache"])[1]
        logits, cache = scanned.apply({**variables, **cache}, ids[:, :6], decode=True, mutable=["cache"])
        for t in range(6, LENGTH):
            logits, cache = scanned.apply({**variables, **cache}, ids[:, t:t + 1], decode=True,
                                          mutable=["cache"])
        return logits[:, -1]

    def last(model):
        return jax.jit(lambda variables, ids: model.apply(variables, ids)[:, -1])

    assert_as_exact(decoded, last(plain), draw_model, "memory decoder decode",
                    seeds=SEEDS["memory decoder decode"], truth_fn=last(wide))


ADAPTED = {"titans": ("titans", ("layers_0", "layers_1"), 2 * HEADS * 2),
           "titans head_0": ("titans", ("head_0.layers_0",), 2),
           "gated_linear_attention": ("gated_linear_attention", ("kernel",), 2 * HEADS),
           "delta_rule": ("delta_rule", ("kernel",), 2 * HEADS)}
"""Each case's configuration, the inner kernels an adapter targets by their module names
(`dew.lora.LoRA.modules`) and how many it targets over both memory layers: Titans' two MLP layers and a
linear memory's one map at every head, and the first layer of the first head alone, the other heads'
networks as they were."""


def factor_draw(model, lora):
    """A draw of a seed's base weights for `model` as its init draws them, the factors of `lora`
    attached over them each drawn a tenth of a normal from zero, and its token rows."""
    ids = jnp.zeros((ROWS, LENGTH), jnp.int32)
    initialize = jax.jit(model.init)
    attached = traverse_util.flatten_dict(lora.apply(model, initialize(jax.random.key(0), ids), key=0)
                                          .variables["params"])

    def draw(seed: int):
        ids = np.random.default_rng([1649, seed]).integers(0, 37, (ROWS, LENGTH), dtype=np.int32)
        base = traverse_util.flatten_dict(initialize(jax.random.key(seed), ids)["params"])
        rng = np.random.default_rng([1650, seed])
        factors = {path: (0.1 * rng.standard_normal(leaf.shape)).astype(np.float32)
                   for path, leaf in attached.items() if path not in base}
        return {"params": traverse_util.unflatten_dict({**base, **factors})}, ids

    return draw


@pytest.mark.parametrize("name", sorted(ADAPTED))
def test_an_adapter_on_a_memorys_inner_kernels_trains_drops_and_merges(name):
    """LoRA on inner kernels of two memory layers (`dew.lora`), every head's or one head's alone: fresh,
    the decoder's logits are the base's bit for bit; its loss's gradient reaches every targeted B
    factor; a forward that carries the dropout stream drops the branches' inputs, so its logits move;
    and with drawn factors the adapted decoder computes its merge, on par against the merge in float64
    (check 1): a retention forgets the adapter's update as it forgets W_0's, and a linear memory's
    recurrence reads the adapted map."""
    from dew.lora import LoRA

    configuration, modules, targets = ADAPTED[name]
    model, wide = decoder(layer(configuration)), decoder(layer(configuration), dtype=jnp.float64)
    lora = LoRA(rank=2, modules=modules, dropout=0.5)
    draw = factor_draw(model, lora)
    ids = jnp.asarray(draw(0)[1])
    base = model.init(jax.random.key(0), ids)
    adapter = lora.apply(model, base, key=0)
    adapted = adapter.model
    np.testing.assert_array_equal(np.asarray(adapted.apply(adapter.variables, ids)),
                                  np.asarray(model.apply(base, ids)))

    gradient = jax.grad(lambda factors: jnp.sum(adapted.apply({**adapter.variables, "params": factors}, ids)
                                                ** 2))(adapter.variables["params"])
    inner = [leaf for path, leaf in traverse_util.flatten_dict(gradient).items()
             if path[-1] == "lora_B" and head_of(path) is not None]
    assert len(inner) == targets
    assert all(np.abs(np.asarray(leaf)).max() > 0 for leaf in inner)

    drawn_factors, ids = draw(0)
    dropped = adapted.apply(drawn_factors, ids, rngs={"dropout": jax.random.key(3)})
    assert not np.array_equal(np.asarray(dropped), np.asarray(adapted.apply(drawn_factors, ids)))

    def merged(model):
        return jax.jit(lambda variables, ids: model.apply(adapter.merge(variables), ids))

    assert_as_exact(jax.jit(adapted.apply), merged(model), draw,
                    f"adapted {name} memory", seeds=SEEDS["adapted memory"], truth_fn=merged(wide))


def test_a_cached_linear_memory_keeps_one_cache_across_dropout_routes():
    """An adapter whose dropout draws on a linear memory's heads takes the memory off the shared
    recurrence on a call carrying the dropout stream (a training forward) and leaves it there on a call
    without, so the two compute by different paths; the memory's decode cache holds one representation
    for both. A decoder's cache allocated without the stream (`init_cache`), then a prefill and a
    token-by-token decode under it, compute bit for bit what the same calls compute from a cache
    allocated under the stream; and a cache allocated under it, continued without it, what one allocated
    without it does."""
    from dew.lora import LoRA
    from dew.objectives.base import training_rngs

    model = decoder(layer("gated_linear_attention"))
    lora = LoRA(rank=2, modules=("kernel",), dropout=0.5)
    variables, ids = factor_draw(model, lora)(0)
    adapted = lora.apply(model, model.init(jax.random.key(0), jnp.asarray(ids)), key=0).model

    @jax.jit
    def cached(allocating, continuing):
        cache = adapted.apply(variables, ROWS, method=adapted.init_cache, mutable=["cache"],
                              rngs=allocating)[1]
        logits, cache = adapted.apply({**variables, **cache}, ids[:, :6], decode=True, mutable=["cache"],
                                      rngs=continuing)
        reads = [logits]
        for t in range(6, LENGTH):
            logits, cache = adapted.apply({**variables, **cache}, ids[:, t:t + 1], decode=True,
                                          mutable=["cache"], rngs=continuing)
            reads.append(logits)
        return jnp.concatenate(reads, axis=1)

    dropped = training_rngs(jax.random.key(5))
    np.testing.assert_array_equal(np.asarray(cached(None, dropped)),
                                  np.asarray(cached(training_rngs(jax.random.key(7)), dropped)))
    np.testing.assert_array_equal(np.asarray(cached(dropped, None)), np.asarray(cached(None, None)))


SCALE = 0.125
"""The FP8 input scale of the quantized checks' inner sites: inputs of a unit's spread take E4M3's
coarse steps at 8 to 16 times it."""


def test_a_cached_linear_memory_off_the_recurrence_continues_its_full_pass():
    """A linear memory whose heads' inputs a checkpoint's input quantization rounds runs its general path
    on every call, while its cache holds the recurrence's state: a prefill and a token-by-token decode
    through that cache read each token as the full pass does, on par against its paper's update over
    the rounded inputs in float64 (check 1)."""
    from dew.quantization import QUANTIZATION, InputQuantization, InputQuantizer

    module = layer("gated_linear_attention")
    draw = drawn(module)
    record = InputQuantization(sites=tuple((site, "fp8") for site in inner_sites(draw(0)[1]["params"])))
    quantized = record.apply(module)
    scales = jax.tree.map(lambda _: np.float32(SCALE), record.declared())

    def scaled(seed: int):
        x, params = draw(seed)
        return x, {**params, QUANTIZATION: scales}

    rounded = InputQuantizer.fp8(SCALE).qdq
    assert_as_exact(decoded(quantized), applied(quantized), scaled,
                    "decoded quantized gated_linear_attention memory",
                    seeds=MEMORY_SEEDS,
                    truth_fn=written(module, dtype=jnp.float64, inputs=rounded))


def level() -> InContextMemory:
    """A continuum-memory level over its block's raw stream, one head of Titans' MLP updated every
    token: its network's input is the stream itself, the same values in every precision."""
    return layer("cms").clone(network=MLPMemory(hidden=(2 * WIDTH,)), commit="token")


def test_a_memorys_inner_network_rounds_its_inputs_at_their_declared_scale():
    """A checkpoint's input quantization at an inner site (`dew.quantization.InputQuantization`) rounds
    that site's input at its scale at every application of the network, the gradient's prediction and
    the readout: the layer computes its paper's update over the inputs E4M3 rounds at the scale its
    variables hold, on par against float64 (check 1)."""
    from dew.quantization import QUANTIZATION, InputQuantization, InputQuantizer

    module = level()
    draw = drawn(module)
    site = next(site for site in inner_sites(draw(0)[1]["params"]) if site.endswith("layers_0"))
    record = InputQuantization(sites=((site, "fp8"),))
    quantized = record.apply(module)
    scales = jax.tree.map(lambda _: np.float32(SCALE), record.declared())

    def scaled(seed: int):
        x, params = draw(seed)
        return x, {**params, QUANTIZATION: scales}

    rounded = InputQuantizer.fp8(SCALE).qdq
    assert_as_exact(applied(quantized), written(module, dtype=jnp.float32, inputs=rounded), scaled,
                    "quantized memory", seeds=MEMORY_SEEDS,
                    truth_fn=written(module, dtype=jnp.float64, inputs=rounded))


def test_a_quantized_memory_decoder_reloads_from_its_record():
    """A decoder computing under input quantization at its memories' first heads' first inner sites, the
    other heads' networks as they were, reads their scales: its logits move from the plain decoder's at
    the same weights; and its record rebuilds a decoder that computes them bit for bit."""
    from dew.config import ModelConfig
    from dew.quantization import QUANTIZATION, InputQuantization
    from dew.registry import to_record

    plain = decoder()
    ids = jax.random.randint(jax.random.key(1), (ROWS, LENGTH), 0, 37)
    first = inner_sites(plain.init(jax.random.key(0), ids)["params"], heads=(0,))
    sites = tuple((site, "fp8") for site in first if site.endswith("layers_0"))
    quantized = InputQuantization(sites=sites).apply(plain)
    variables = quantized.init(jax.random.key(0), ids)
    scales = jax.tree.map(lambda scale: np.float32(SCALE), variables[QUANTIZATION])
    variables = {**variables, QUANTIZATION: scales}
    logits = np.asarray(quantized.apply(variables, ids))
    assert not np.array_equal(logits, np.asarray(plain.apply({"params": variables["params"]}, ids)))
    record = json.loads(json.dumps(to_record(ModelConfig.from_model(quantized), ModelConfig)))
    reloaded = ModelConfig.from_dict(record).build()
    np.testing.assert_array_equal(np.asarray(reloaded.apply(variables, ids)), logits)


FROZEN = {"titans": layer("titans"), "linear": layer("delta_rule").clone(momentum=Gate(value=0.0))}
"""The frozen FP8 check's memories: Titans' biased MLP, whose first layer's kernel alone each head's
packing takes, and a linear memory with a momentum of zero (off the recurrence, a general path as the
decoded layer's), whose one kernel is all a head holds, so the packing takes every one of its
parameters."""


@pytest.mark.parametrize("name", sorted(FROZEN))
def test_a_frozen_fp8_inner_kernel_is_w0_as_its_codes_decode(name):
    """An inner kernel a frozen FP8 record packs (`InputQuantization.frozen_fp8`, a LoRA run's FP8 base)
    is the memory's W_0 as its codes decode (`frozen_fp8_kernel`, as a merge decodes it), which the
    in-context steps move from, a head packed whole holding no parameter of its own: the layer computes,
    bit for bit, the same layer whose kernels are those decoded ones."""
    from dew.quantization import QUANTIZATION, InputQuantization, frozen_fp8_kernel

    module = FROZEN[name]
    x, params = drawn(module)(0)
    sites = tuple(site for site in inner_sites(params["params"])
                  if name != "titans" or site.endswith("layers_0"))
    record = InputQuantization(frozen_fp8=sites, measured_on="sm90")
    codes = record.packed(params["params"])
    kernels = {}
    for site in sites:
        leaves = codes
        for scope in site.split("/"):
            leaves = leaves[scope]
        kernels[(*site.split("/"), "kernel")] = frozen_fp8_kernel(leaves)
    flat = traverse_util.flatten_dict(params["params"])
    held = traverse_util.unflatten_dict({path: leaf for path, leaf in flat.items() if path not in kernels})
    frozen = jax.jit(record.apply(module).apply)({"params": held, QUANTIZATION: codes}, x)
    decoded = jax.jit(module.apply)({"params": traverse_util.unflatten_dict({**flat, **kernels})}, x)
    np.testing.assert_array_equal(np.asarray(frozen), np.asarray(decoded))


def explicit_rows_run(module: InContextMemory):
    """`module`'s forward over whole rows, the gradients of its weights and stream under a cotangent,
    its forward over rows packing two documents (`segments`), and a prefill of six tokens continued one
    a cached call (`decoded`)."""
    cached = decoded(module)

    @jax.jit
    def run(x, params, cotangent, segments):
        full, pull = jax.vjp(lambda held, x: module.apply(held, x), params, x)
        laid = coordinates(segments=segments, max_segments=2)
        call = MixerCall(metadata=AttentionMetadata(coordinates=laid))
        return full, pull(cotangent), module.apply(params, x, call), cached(x, params)

    return run


MESHED = {"titans": layer("titans"), "titans committed": layer("titans").clone(commit="chunk"),
          "gated_linear_attention": layer("gated_linear_attention"),
          "cms": layer("cms").clone(normalize=True, learning_rate=Gate(scale=STEP))}
"""The memories the mesh check splits: Titans' each token reading its own step or its chunk's start,
a linear memory on the recurrence, and a continuum-memory level, an existing gated MLP over the stream
committing per chunk, its keys and queries normalized as the papers' and at their step (over the raw
stream, at the level's own hundredth or the papers' step, its weights' gradients do not resolve within
512 seeds)."""


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize(("name", "axis"), [("titans", "Explicit"), ("titans committed", "Explicit"),
                                            ("gated_linear_attention", "Explicit"), ("cms", "Explicit"),
                                            ("cms", "Auto")])
def test_a_memory_runs_its_rows_split_over_a_data_axis(name, axis):
    """Rows split over a data axis, Explicit or Auto, as a run's batch is: a memory's forward, the
    gradients of its weights and stream, its forward over packed documents and a prefill continued token
    by token through its cache, each on the general chunk scan (each token reading its own step, or its
    chunk's start), the token scan or a linear memory's recurrence, are what one device computes, on
    par against float64 (check 1). On an Auto mesh a gated MLP's activations are placed per token as its
    rank names them."""
    from jax.sharding import AxisType, NamedSharding, PartitionSpec

    module = MESHED[name]
    mesh = jax.make_mesh((2,), ("data",), axis_types=(getattr(AxisType, axis),), devices=jax.devices()[:2])
    run, draw = explicit_rows_run(module), drawn(module)
    segments = np.broadcast_to(np.where(np.arange(LENGTH) < 5, 1, 2), (ROWS, LENGTH)).astype(np.int32)

    def placed(seed: int):
        x, params = draw(seed)
        cotangent = np.random.default_rng([1651, seed]).standard_normal((*x.shape[:2], WIDTH))
        return x, params, cotangent.astype(np.float32), segments

    @as_given
    def split(x, params, cotangent, segments):
        with jax.set_mesh(mesh):
            rows, whole = NamedSharding(mesh, PartitionSpec("data")), NamedSharding(mesh, PartitionSpec())
            return jax.device_get(run(jax.device_put(x, rows), jax.device_put(params, whole),
                                      jax.device_put(cotangent, rows), jax.device_put(segments, rows)))

    assert_as_exact(split, run, placed, f"{name} memory rows split over {axis}",
                    seeds=SEEDS["rows split cms" if name == "cms" else "rows split"],
                    truth_fn=explicit_rows_run(module.clone(dtype=jnp.float64)))


INNER = r"(.*/)?head_\d+/.*"
"""A Qwix pattern selecting every inner Dense of a memory's heads' networks."""


def test_a_quantized_memorys_update_rounds_from_its_own_stream():
    """Qwix's fake-quantized training (`dew.quantization.Quantization`) of every inner Dense of two
    memory layers, its gradients int8 and rounded stochastically: each application of a head's network,
    the memory's inner gradients among them, draws from its token's key of the call's own
    `stochastic_rounding` stream (`training_rngs`), so the same key repeats the logits and their outer
    gradients bit for bit, another moves them, and both are finite."""
    pytest.importorskip("qwix")
    from dew.objectives.base import training_rngs
    from dew.quantization import Quantization

    model = Quantization(patterns=(INNER,), bwd_qtype="int8", bwd_stochastic_rounding="uniform").apply(
        decoder())
    ids = jax.random.randint(jax.random.key(1), (ROWS, LENGTH), 0, 37)
    params = model.init(jax.random.key(0), ids)["params"]

    @jax.jit
    def trained(params, key):
        rngs = training_rngs(key)
        logits, pull = jax.vjp(lambda held: model.apply({"params": held}, ids, rngs=rngs), params)
        return logits, pull(jnp.ones_like(logits))[0]

    drawn = trained(params, jax.random.key(0))
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(drawn))
    jax.tree.map(np.testing.assert_array_equal, drawn, trained(params, jax.random.key(0)))
    other = trained(params, jax.random.key(1))
    assert not np.array_equal(np.asarray(drawn[0]), np.asarray(other[0]))


class ConvolvedMemory(nn.Module):
    """An inner network that convolves the view's vector as a one-channel signal, width kept: a memory's
    network that runs a convolution rather than a matmul."""

    dtype: jnp.dtype | None = None

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        return nn.Conv(1, (3,), padding="SAME", dtype=self.dtype, name="conv")(x[..., None])[..., 0]


SERVED = {"moving": ((INNER,), None, True), "projections": ((r".*self_attn/[qkvo]_proj",), None, False),
          "convolved": ((INNER,), ConvolvedMemory(), True)}
"""The served-memory cases: the patterns serving quantizes, the memory's inner network (None: Titans'
MLP), and whether those weights move."""


def served_site(case: str):
    """The served-memory site `case` (`SERVED`): the served decoder over its stored tree, from seed 0's
    serving preparation, and check 1's arms over each seed's: the served decoder, quantized training
    from the stored W_0, and that training in float64, with the draw."""
    from model_support import model_draw

    from dew.quantization import Quantization, decoded_stored, quantize_for_serving

    patterns, network, moving = SERVED[case]
    memory = None if network is None else InContextMemory(
        network=network, heads=2, head_features=8, chunk=4, normalize=True,
        learning_rate=Gate(scale=STEP), momentum=Gate(), retention=Gate(bias=-2.0))
    model, wide = decoder(memory), decoder(memory, dtype=jnp.float64)
    spec = Quantization(patterns=patterns)
    draw_model = model_draw(model, (ROWS, LENGTH), 37)
    served = quantize_for_serving(model, *draw_model(0)[:1], spec, draw_model(0)[1])[0]

    def draw(seed: int):
        variables, ids = draw_model(seed)
        stored = quantize_for_serving(model, variables, spec, ids)[1]
        start = {**variables, "params": decoded_stored(stored)["params"]} if moving else variables
        return stored, start, ids

    return (served, lambda stored, start, ids: served.apply(stored, ids),
            lambda stored, start, ids: spec.apply(model).apply(start, ids), draw,
            lambda stored, start, ids: spec.apply(wide).apply(start, ids))


@pytest.mark.parametrize("case", SERVED)
def test_a_served_memory_computes_the_quantized_memory_from_its_stored_w0(case):
    """Serving quantization (`quantize_for_serving`) of two memory layers' inner networks stores each
    head's W_0 quantized, the steps starting from the float it decodes to (`decoded_stored`), and rounds
    every product of the moving weights, a matmul's or a convolution's, as quantized training does
    (`moving_weights`): the served decoder computes what quantized training (`Quantization.apply`)
    computes from the stored W_0, on par against that training in float64 (check 1). Its projections,
    which no step moves, serve as quantized training computes them from their float weights. The same
    tree after a round trip through the host computes the same program bit for bit."""
    pytest.importorskip("qwix")
    from dew.quantization import Annotated

    served, dew_fn, reference_fn, draw, truth_fn = served_site(case)
    held, _, ids = draw(0)
    assert any(isinstance(leaf, Annotated)
               for leaf in jax.tree.leaves(held, is_leaf=lambda leaf: isinstance(leaf, Annotated)))
    reloaded = jax.tree.map(jnp.asarray, jax.device_get(held))
    np.testing.assert_array_equal(np.asarray(served.apply(reloaded, ids)),
                                  np.asarray(served.apply(held, ids)))
    assert_as_exact(dew_fn, reference_fn, draw, f"served memory {case}", seeds=SEEDS[f"served memory {case}"],
                    truth_fn=truth_fn)


def test_a_served_memory_rounds_its_moving_gradients_from_its_requests_stream():
    """A memory decoder served quantized (`TextGeneration.quantized`) under a spec that rounds the
    backward stochastically (`bwd_qtype`, `bwd_stochastic_rounding`) draws its inner steps' rounding from
    its request's own key, folded with each token's place (`dew.nn.kv_cache.ROUNDING_KEYS`), apart from
    its sampling: a request generates alike alone and served beside another, in either slot of one live
    server, tokens, lengths, termination and likelihoods; and another key rounds apart, its likelihoods
    moving though greedy decoding draws nothing. Without the stream the decode raised in Qwix's make_rng
    (ledger 2020)."""
    pytest.importorskip("qwix")
    from model_support import Digits
    from test_serving import assert_same_generation

    from dew.inference import RunProcessor, TextGeneration
    from dew.inference.serving import Server
    from dew.quantization import Quantization
    from dew.sampling import Sampling

    model = decoder().clone(max_seq_len=64)
    params = model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))
    task = TextGeneration(model, params, RunProcessor(Digits()),
                          sampling=Sampling(temperature=0, eos_token_ids=12))
    bound = task.quantized(Quantization(patterns=(INNER,), bwd_qtype="int8",
                                        bwd_stochastic_rounding="uniform"))
    alone = {prompt: bound(prompt, 4, key=3) for prompt in ("12", "567")}
    server = Server.from_task(bound, slots=2, capacity=64)
    for prompts in (["12", "567"], ["567", "12"]):
        for served, prompt in zip(server(prompts, 4, key=3), prompts, strict=True):
            assert_same_generation(served, alone[prompt])
    assert not np.array_equal(np.asarray(bound("12", 4, key=4).raw_log_probs),
                              np.asarray(alone["12"].raw_log_probs))


def test_a_qwix_quantized_linear_memory_leaves_the_recurrence():
    """A linear memory whose inner kernel Qwix fake-quantizes (`dew.quantization.Quantization`) computes
    its network as the quantized product, no linear map of its input by its kernel, which the shared
    recurrence's rows are, so it runs its general path: the delta rule computes, bit for bit, the same
    quantized network with a momentum of zero, which no recurrence row takes."""
    pytest.importorskip("qwix")
    from dew.quantization import Quantization

    spec = Quantization(patterns=(INNER,))
    x, params = drawn(layer("delta_rule"))(0)
    routed = spec.apply(layer("delta_rule")).apply(params, x)
    general = spec.apply(layer("delta_rule").clone(momentum=Gate(value=0.0))).apply(params, x)
    np.testing.assert_array_equal(np.asarray(routed), np.asarray(general))


def test_a_memory_composes_under_a_field_named_network():
    """A memory held in a module's field named `network`, as a public wrapper holds the module it runs,
    initializes and computes as the memory alone over the same weights."""
    class Holder(nn.Module):
        network: InContextMemory

        def __call__(self, x):
            return self.network(x)

    module = layer("titans")
    x = drawn(module)(0)[0]
    variables = Holder(module).init(jax.random.key(0), x)
    np.testing.assert_array_equal(np.asarray(Holder(module).apply(variables, x)),
                                  np.asarray(module.apply({"params": variables["params"]["network"]}, x)))


def test_a_memory_refuses_a_bidirectional_decoder():
    """A memory reads each token's prefix alone, so a decoder that asks its layers for the whole row
    (`causal=False`, a masked diffusion model's) refuses to build one; the same decoder causal builds."""
    ids = jnp.zeros((2, LENGTH), jnp.int32)
    decoder().init(jax.random.key(0), ids)
    with pytest.raises(ValueError):
        decoder(causal=False).init(jax.random.key(0), ids)


def test_a_memory_decoder_marks_and_climbs_by_the_paths_it_can_take(capsys):
    """A memory kind marks (`MixerBase.marks`) the residuals of every path its configuration can take,
    since a wrapper the kind cannot see takes a linear memory off the recurrence: each name a decoder of
    the kind marks, as a policy saving that name alone keeps for the backward more than a policy saving
    nothing, is among them, wrapped or not, and the decoder's remat ladder skips by them. A linear
    memory on the delta rule marks the rule's port's decay, inverse and memory on the recurrence, and
    its chunk scan's carry under an input quantization of its first head, which takes it off the
    recurrence; its ladder takes the rung recomputing the rule's memory alone. Titans' memory marks its
    carry, and its ladder has no such rung."""
    from jax.ad_checkpoint import print_saved_residuals
    from model_support import TINY_DECODER

    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.backbones.decoder_block import RematPolicy
    from dew.nn.kernels.generation import chosen
    from dew.nn.mixer_base import DELTA_RESIDUALS, MEMORY_RESIDUALS
    from dew.nn.mixers.memory import InContextMemoryMixer
    from dew.quantization import QUANTIZATION, InputQuantization

    ids = jax.random.randint(jax.random.key(3), (2, LENGTH), 0, 37)

    def kept(memory: InContextMemory, *, quantized: bool = False) -> tuple[set[str], list]:
        kind = InContextMemoryMixer(memory)
        model = CausalTransformer(**{"vocab_size": 37, **TINY_DECODER, "dtype": jnp.float32, "mixer": kind})
        variables = model.init(jax.random.key(4), ids)
        if quantized:
            sites = tuple((site, "fp8") for site in inner_sites(variables["params"], heads=(0,)))
            record = InputQuantization(sites=sites)
            model = record.apply(model)
            scales = jax.tree.map(lambda _: np.float32(SCALE), record.declared())
            variables = {**variables, QUANTIZATION: scales}

        def saved(names: tuple[str, ...]) -> str:
            capsys.readouterr()
            loss = jax.checkpoint(lambda held: jnp.sum(model.apply(held, ids) ** 2),
                                  policy=RematPolicy(save=names).checkpoint_policy())
            print_saved_residuals(loss, variables)
            return capsys.readouterr().out

        nothing = saved(())
        marked = {name for name in (*DELTA_RESIDUALS, *MEMORY_RESIDUALS) if saved((name,)) != nothing}
        assert marked and marked <= set(kind.marks())
        climbed = [model]
        while (stronger := climbed[-1].recompute_more()) is not None:
            climbed.append(stronger)
        return marked, [rung.recompute_record() for rung in climbed]

    with chosen(gated_delta_rule="fla"):
        delta, delta_rungs = kept(layer("delta_rule"))
        quantized, _ = kept(layer("delta_rule"), quantized=True)
    titans, titans_rungs = kept(layer("titans"))
    assert set(DELTA_RESIDUALS) <= delta and "all_but_delta_memory" in delta_rungs
    assert CARRY in quantized
    assert CARRY in titans and "all_but_delta_memory" not in titans_rungs


def test_muon_steps_each_memory_heads_matrices_on_their_own():
    """A memory decoder trains under Muon: each head's inner matrices are matrices Muon orthogonalizes on
    their own, by Linen's kernel convention (inputs into outputs), its biases take AdamW's update, and
    one step of the optimizer runs over the whole tree."""
    import optax

    from dew.optim import OptimConfig, muon_weight_dimension_numbers

    model = decoder()
    params = model.init(jax.random.key(0), jnp.zeros((2, LENGTH), jnp.int32))["params"]
    numbers = muon_weight_dimension_numbers(params)["layers_0"]["self_attn"]
    layers = {f"layers_{index}": {"kernel": optax.contrib.MuonDimensionNumbers(), "bias": None}
              for index in (0, 1)}
    assert {name: tree for name, tree in numbers.items() if head_of((name,)) is not None} == {
        f"head_{head}": layers for head in range(2)}
    solver = OptimConfig(optimizer="muon", learning_rate=1e-3).build(10)
    updates, _ = solver.update(jax.tree.map(jnp.ones_like, params), solver.init(params), params)
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(updates))


def test_a_memory_layer_kind_records_and_rebuilds():
    """A memory layer kind records the memory's whole configuration: a decoder whose two kinds set every
    field a memory computes by away from its default (heads, an existing GatedMLP as a level's network
    over the raw stream with no output projection, an lp and a Huber objective, a self-generated
    target, constant and scaled gates, momentum, retention, Newton-Schulz steps, the chunk and its
    commit, weights the inner loop holds at W_0) writes a JSON record that rebuilds the same model,
    which computes the original's logits bit for bit."""
    from model_support import TINY_DECODER

    from dew.config import ModelConfig
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.backbones.layer_plan import LayerKind
    from dew.nn.mixers.memory import InContextMemoryMixer
    from dew.registry import to_record

    heads = InContextMemory(network=MLPMemory(hidden=(24,), activation="gelu"), heads=4, head_features=8,
                            objective=Objective("lp", p=3.0), target="self_generated",
                            learning_rate=Gate(scale=0.05, bias=0.5), momentum=Gate(value=0.5),
                            retention=Gate(scale=0.5, bias=-1.0), chunk=2, orthogonalize=3, normalize=True)
    level = InContextMemory(network=GatedMLP(hidden_features=48, out_features=32), project_query=False,
                            project_key=False, project_value=False, output=False,
                            objective=Objective("huber", delta=0.5), learning_rate=Gate(value=0.01), chunk=3,
                            commit="chunk", slow=("gate_proj",))
    model = CausalTransformer(**{**TINY_DECODER, "vocab_size": 37, "num_layers": 3, "dtype": jnp.float32,
                                 "layer_types": ("heads", "level", "attention"),
                                 "kinds": {"heads": LayerKind(mixer=InContextMemoryMixer(heads)),
                                           "level": LayerKind(mixer=InContextMemoryMixer(level)),
                                           "attention": LayerKind()}})
    record = json.loads(json.dumps(to_record(ModelConfig.from_model(model), ModelConfig)))
    rebuilt = ModelConfig.from_dict(record).build()
    assert rebuilt == model
    ids = jax.random.randint(jax.random.key(1), (2, LENGTH), 0, 37)
    variables = model.init(jax.random.key(0), ids)
    np.testing.assert_array_equal(np.asarray(rebuilt.apply(variables, ids)),
                                  np.asarray(model.apply(variables, ids)))


def test_a_memory_draws_its_tokens_keys_from_a_typed_rbg_stream():
    """A training forward whose streams are typed rbg keys, not the default threefry: the memory's
    tokens' keys are rewrapped in their stream's own implementation on both its chunked prefill and its
    token-by-token recurrent readout (ledger 1876 NN-01), so the adapter's dropout draws on them, the
    same key drawing the same logits and another key others."""
    from dew.lora import LoRA
    from dew.objectives.base import training_rngs

    model = decoder(layer("gated_linear_attention"))
    lora = LoRA(rank=2, modules=("kernel",), dropout=0.5)
    variables, ids = factor_draw(model, lora)(0)
    adapted = lora.apply(model, model.init(jax.random.key(0), jnp.asarray(ids)), key=0).model

    @jax.jit
    def cached(rngs):
        cache = adapted.apply(variables, ROWS, method=adapted.init_cache, mutable=["cache"])[1]
        logits, cache = adapted.apply({**variables, **cache}, ids[:, :6], decode=True, mutable=["cache"],
                                      rngs=rngs)
        reads = [logits]
        for t in range(6, LENGTH):
            logits, cache = adapted.apply({**variables, **cache}, ids[:, t:t + 1], decode=True,
                                          mutable=["cache"], rngs=rngs)
            reads.append(logits)
        return jnp.concatenate(reads, axis=1)

    first, again, other = (np.asarray(cached(training_rngs(jax.random.key(seed, impl="rbg"))))
                           for seed in (5, 5, 6))
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)
