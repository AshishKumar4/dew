"""Dropless exchange and capacity dropping against the global sort/gather path on CPU.

Capacity dropping is also held to MaxText's own `generate_masks`
(tests/fixtures/maxtext/capacity.npz, tools/maxtext_capacity_reference.py)
and the layer to a float64 expert loop over the slots it keeps.
"""

import functools
import json
import math
from pathlib import Path

import jax
import jax.extend
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from jax.sharding import NamedSharding, PartitionSpec as P
from moe_support import assert_exchange_across_processes, objective, routing_case
from reference_error import as_given, assert_as_exact, assert_computes_the_oracle

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import Mixture
from dew.nn.moe import ExpertMLP, capacity_positions
from dew.training import MeshSpec

MAXTEXT = Path(__file__).resolve().parent / "fixtures" / "maxtext" / "capacity.npz"

pytestmark = pytest.mark.mesh





@pytest.mark.parametrize("shards,fsdp,tokens,top_k,skewed,scale_inputs,activation,limit", [
    (2, 4, 24, 3, False, False, 'swiglu', None),
    (4, 2, 24, 2, True, False, 'swiglu', None),
    (8, 1, 24, 3, False, True, 'geglu_exact', 0.7),
    (4, 2, 7, 2, True, True, 'geglu', None),
    (4, 1, 1, 1, True, False, 'swiglu', None),
    (2, 1, 0, 2, True, False, 'swiglu', None),
])
def test_exchange_retains_outputs_and_all_differentiable_inputs(
        shards, fsdp, tokens, top_k, skewed, scale_inputs, activation, limit):
    x, weights, indices = map(jnp.asarray, routing_case(tokens, 8, top_k, skewed))
    reference = ExpertMLP(8, 12, 8, scale_inputs=scale_inputs, activation=activation,
                          swiglu_limit=limit)
    parameters = nn.unbox(reference.init(jax.random.key(19), x, weights, indices))
    expected = jax.jit(jax.value_and_grad(functools.partial(objective, reference, indices),
                                        (0, 1, 2), has_aux=True))(parameters, x, weights)
    mesh = MeshSpec(expert=shards, fsdp=fsdp).build()
    with jax.set_mesh(mesh):
        actual = jax.jit(jax.value_and_grad(functools.partial(
            objective, reference.clone(dispatch='exchange'), indices),
            (0, 1, 2), has_aux=True))(parameters, x, weights)
    # Chunking changes the order of fp32 expert-weight cotangent sums.
    for actual_leaf, expected_leaf in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(actual_leaf.astype(jnp.float32), expected_leaf.astype(jnp.float32),
                                   atol=3e-5, rtol=3e-5)
    if skewed:
        for projection in actual[1][0]['params'].values():
            np.testing.assert_array_equal(projection['kernel'][top_k:], 0)


def primitives(jaxpr: jax.extend.core.Jaxpr) -> list[str]:
    """Every primitive `jaxpr` runs, its sub-jaxprs' (a shard_map's body, a scan's) included."""
    names = []
    for equation in jaxpr.eqns:
        names.append(equation.primitive.name)
        for value in equation.params.values():
            for held in value if isinstance(value, (tuple, list)) else (value,):
                if isinstance(held, jax.extend.core.ClosedJaxpr):
                    names += primitives(held.jaxpr)
                elif isinstance(held, jax.extend.core.Jaxpr):
                    names += primitives(held)
    return names


@pytest.mark.parametrize("skewed", [False, True], ids=["even", "skewed"])
def test_a_dropless_exchange_moves_its_rows_in_one_ragged_all_to_all_each_way(skewed):
    """An expert mesh's dropless exchange sends each shard its rows in one `lax.ragged_all_to_all` and
    takes them back in one more, the collective XLA runs, whatever the routing's skew: no rounds of
    equal buckets, no loop over them (ledger 1711); its gradient moves its rows by the same collective."""
    x, weights, indices = map(jnp.asarray, routing_case(24, 8, 2, skewed))
    layer = ExpertMLP(8, 12, 8).clone(dispatch='exchange')
    parameters = nn.unbox(ExpertMLP(8, 12, 8).init(jax.random.key(19), x, weights, indices))
    with jax.set_mesh(MeshSpec(expert=4, fsdp=2).build()):
        forward = primitives(jax.make_jaxpr(layer.apply)(parameters, x, weights, indices).jaxpr)
        backward = primitives(jax.make_jaxpr(jax.grad(functools.partial(objective, layer, indices),
                                                      has_aux=True))(parameters, x, weights).jaxpr)
    assert forward.count("ragged_all_to_all") == 2 and "all_to_all" not in forward and "scan" not in forward
    # JAX's transpose of the collective exchanges its offsets by an all_to_all of its own.
    assert "ragged_all_to_all" in backward and "scan" not in backward


@pytest.mark.parametrize("spec, exchanged", [
    (MeshSpec(expert=4, fsdp=2), True), (MeshSpec(expert=2, fsdp=4), True), (MeshSpec(fsdp=8), False)],
    ids=["4-way", "2-way", "no-expert-axis"])
def test_an_expert_mesh_exchanges_by_default_and_any_other_mesh_dispatches_globally(spec, exchanged):
    """A mixture's default dispatch (`'auto'`) keeps an expert mesh's experts in place, exchanging the
    tokens (`ragged_all_to_all`), and gathers no expert's weights; on a mesh without an expert axis it
    dispatches globally, as `'global'` does (ledger 1711: on a 4-way expert mesh, the global dispatch
    gathered every expert's weights, 200,704 bytes of eight, where the exchange moved 4,096)."""
    x, weights, indices = map(jnp.asarray, routing_case(24, 8, 2, skewed=False))
    layer = ExpertMLP(8, 12, 8)
    parameters = nn.unbox(layer.init(jax.random.key(19), x, weights, indices))
    with jax.set_mesh(spec.build()):
        ran = primitives(jax.make_jaxpr(layer.apply)(parameters, x, weights, indices).jaxpr)
        named = layer.clone(dispatch="exchange" if exchanged else "global")
        named = primitives(jax.make_jaxpr(named.apply)(parameters, x, weights, indices).jaxpr)
    assert ("ragged_all_to_all" in ran) == exchanged
    assert ran == named


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize("expert, exchanged", [(4, True), (1, False)], ids=["expert-mesh", "no-expert-axis"])
def test_a_config_built_runs_step_exchanges_on_an_expert_mesh_by_default(tmp_path, expert, exchanged):
    """An LM run built from its config, a mixture of eight experts at its default dispatch, lowers its
    training step (`--trainer.lower-for`) with the exchange's `ragged_all_to_all` on a 4-way expert mesh
    and with none where the mesh has no expert axis (ledger 1711's flow, row 4615)."""
    import dataclasses

    from run_kinds import lm, trained

    from dew.training._lowering import Lowered

    run = lm(tmp_path, drawn=False)
    fields = {**run.model.fields, "mixture": {"experts": 8, "top_k": 2, "expert_features": 16}}
    run = dataclasses.replace(run, model=dataclasses.replace(run.model, fields=fields))
    with pytest.raises(Lowered) as lowered:
        trained(run, tmp_path, "lowered", trainer={"mesh": MeshSpec(expert=expert, fsdp=8 // expert),
                                                   "lower_for": "sm90"})
    assert ("ragged_all_to_all" in lowered.value.lowered.as_text()) == exchanged


def test_exchange_preserves_placed_expert_and_width_gradients():
    x, weights, indices = map(jnp.asarray, routing_case(24, 8, 2, skewed=False))
    model = ExpertMLP(8, 16, 8)
    parameters = nn.unbox(model.init(jax.random.key(19), x, weights, indices))
    mesh = MeshSpec(expert=4, fsdp=2).build()
    columns = NamedSharding(mesh, P('expert', None, 'fsdp'))
    parameter_specs = {'params': {
        'gate_proj': {'kernel': columns}, 'up_proj': {'kernel': columns},
        'down_proj': {'kernel': NamedSharding(mesh, P('expert', 'fsdp'))}}}
    token_spec = NamedSharding(mesh, P(('expert', 'fsdp')))
    parameters = jax.device_put(parameters, parameter_specs)
    x, weights, indices = (jax.device_put(value, token_spec) for value in (x, weights, indices))
    specs = ((NamedSharding(mesh, P()), token_spec), (parameter_specs, token_spec, token_spec))
    with jax.set_mesh(mesh):
        # The global dispatch named: the default exchanges on this expert mesh.
        named = model.clone(dispatch='global')
        expected = jax.jit(jax.value_and_grad(functools.partial(objective, named, indices),
                            (0, 1, 2), has_aux=True), out_shardings=specs)(parameters, x, weights)
        actual = jax.jit(jax.value_and_grad(functools.partial(
            objective, model.clone(dispatch='exchange'), indices),
            (0, 1, 2), has_aux=True), out_shardings=specs)(parameters, x, weights)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, atol=3e-5, rtol=3e-5)



@pytest.mark.parametrize("parallel", [False, True])
def test_decoder_exchange_retains_logits_and_expert_bias_observations(parallel):
    mixture = Mixture(experts=4, top_k=2, parallel=parallel, bias=not parallel,
                      shared_features=0 if parallel else 12)
    reference = CausalTransformer(vocab_size=16, emb_features=8, num_heads=2,
                                  num_layers=1, mlp_features=12, max_seq_len=4,
                                  qk_norm=False, mixture=mixture)
    tokens = jnp.arange(16, dtype=jnp.int32).reshape(4, 4)
    parameters = nn.unbox(reference.init(jax.random.key(10), tokens))
    expected, observations = reference.apply(parameters, tokens, mutable=['router'])
    from dataclasses import replace
    exchanged = reference.clone(mixture=replace(mixture, dispatch='exchange'))
    mesh = MeshSpec(expert=2).build()
    with jax.set_mesh(mesh):
        actual, actual_observations = jax.jit(lambda p, t: exchanged.apply(
            p, t, mutable=['router']))(parameters, tokens)
    np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5)
    for actual_leaf, expected_leaf in zip(jax.tree.leaves(actual_observations),
                                          jax.tree.leaves(observations), strict=True):
        np.testing.assert_allclose(actual_leaf, expected_leaf, atol=1e-7, rtol=1e-6)


def maxtext_positions(indices: np.ndarray, experts: int) -> np.ndarray:
    """Each slot's place in its expert's queue as MaxText's `generate_masks`
    counts it: a running count over a sequence's tokens and, within a
    token, over its choices in order."""
    *_, length, top_k = indices.shape
    flat = indices.reshape(-1, length * top_k)
    running = np.cumsum(np.eye(experts, dtype=np.int64)[flat], axis=1)
    return (np.take_along_axis(running, flat[..., None], axis=-1)[..., 0] - 1).reshape(indices.shape)


def maxtext_capacity(length: int, top_k: int, experts: int, factor: float) -> int:
    return int(max(math.ceil(length * top_k / experts) * factor, factor))


@pytest.mark.mesh(devices=4)
@pytest.mark.parametrize('spec,batch,length,top_k', [
    (MeshSpec(expert=2, fsdp=2), 4, 12, 2),
    (MeshSpec(expert=2), 16, 4, 2),
    (MeshSpec(expert=4), 2, 12, 2),
    (MeshSpec(expert=4), 3, 5, 1),
], ids=['whole-sequences', 'two-sequences-a-device', 'part-of-a-sequence', 'padded-straddling'])
@pytest.mark.parametrize('dispatch', ['global', 'exchange'])
def test_capacity_drops_the_same_slots_on_every_layout(spec, batch, length, top_k, dispatch):
    """With a capacity, a layer computes what the dropless layer computes
    with the dropped slots' weights zeroed: outputs, parameter and token
    gradients, and the kept slots' weight gradients. A dropped slot's
    weight has none. The cases split sequences across devices, put several
    on one device, and pad a batch the devices do not divide."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=(batch, length, 8)).astype(np.float32)
    weights = rng.uniform(0.1, 0.9, size=(batch, length, top_k)).astype(np.float32)
    scores = rng.normal(size=(batch, length, 8))
    scores[..., :2] += 1.5
    indices = np.argsort(-scores, axis=-1)[..., :top_k].astype(np.int32)
    kept = maxtext_positions(indices, 8) < maxtext_capacity(length, top_k, 8, 1.0)
    assert not kept.all()
    routing = jnp.asarray(indices)
    dropless = ExpertMLP(8, 12, 8)
    parameters = nn.unbox(dropless.init(jax.random.key(19), x, weights, routing))
    gradient = functools.partial(jax.value_and_grad, argnums=(0, 1, 2), has_aux=True)
    expected = jax.jit(gradient(functools.partial(objective, dropless, routing)))(
        parameters, x, np.where(kept, weights, 0).astype(np.float32))
    dropping = dropless.clone(dispatch=dispatch, capacity_factor=1.0)
    with jax.set_mesh(spec.build(jax.devices()[:4])):
        actual = jax.jit(gradient(functools.partial(objective, dropping, routing)))(
            parameters, x, weights)
    (value, out), (d_parameters, d_x, d_weights) = actual
    (expected_value, expected_out), (expected_parameters, expected_x, expected_weights) = expected
    for a, b in zip(jax.tree.leaves((value, out, d_parameters, d_x)),
                    jax.tree.leaves((expected_value, expected_out, expected_parameters, expected_x)),
                    strict=True):
        np.testing.assert_allclose(a, b, atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(np.where(kept, d_weights, 0), np.where(kept, expected_weights, 0),
                               atol=3e-5, rtol=3e-5)
    np.testing.assert_array_equal(np.where(kept, 0, d_weights), 0)


@pytest.fixture(scope="module")
def maxtext():
    with np.load(MAXTEXT) as loaded:
        arrays = dict(loaded)
    return arrays, json.loads(arrays.pop("meta").tobytes())["cases"]


def maxtext_kept(arrays, case: str) -> np.ndarray:
    """The slots MaxText's dispatch mask keeps, `[batch, length, top_k]`."""
    kept = arrays[f"{case}/dispatch"].any(-1)  # [batch, length, experts]
    return np.take_along_axis(kept, arrays[f"{case}/indices"], axis=-1)


@pytest.mark.mesh(devices=1)
@pytest.mark.parametrize("case", ["whole-sequences", "short", "loose", "tight"])
def test_capacity_keeps_the_slots_maxtext_keeps(maxtext, case):
    """Each slot's queue place and the capacity against MaxText's own
    `generate_masks`: the slots kept, and the capacity its masks size."""
    arrays, cases = maxtext
    shape = cases[case]
    positions, capacity = capacity_positions(jnp.asarray(arrays[f"{case}/indices"]), shape["experts"],
                                             shape["factor"])
    assert capacity == arrays[f"{case}/dispatch"].shape[-1]
    np.testing.assert_array_equal(np.asarray(positions) < capacity, maxtext_kept(arrays, case))


def expert_loop(x, weights, indices, kept, parameters):
    """The routed experts slot by slot: each kept slot's swiglu expert at its
    weight, summed per token (MaxText's combine), apart from Dew's dispatch."""
    kernels = parameters["params"]
    gate, up, down = (kernels[name]["kernel"][indices] for name in ("gate_proj", "up_proj", "down_proj"))
    hidden = jax.nn.silu(jnp.einsum("bsm,bskmh->bskh", x, gate)) * jnp.einsum("bsm,bskmh->bskh", x, up)
    slots = jnp.einsum("bskh,bskhm->bskm", hidden, down)
    return jnp.einsum("bskm,bsk->bsm", slots, weights * kept)


CAPACITY_SEEDS = 16
"""The dropping layer's declared check-1 seeds against the dropless one, sized by `seeds_for` paired on
armada: 8 for every case and mesh but the tight case's global dispatch on one of them, 16."""


@pytest.mark.mesh(devices=4)
@pytest.mark.parametrize("spec", [MeshSpec(expert=2, fsdp=2), MeshSpec(expert=4)],
                         ids=["expert2-fsdp2", "expert4"])
@pytest.mark.parametrize("dispatch", ["global", "exchange"])
@pytest.mark.parametrize("case", ["whole-sequences", "short", "tight"])
def test_capacity_drops_compute_maxtexts_experts(maxtext, case, dispatch, spec):
    """The dropping layer, output and gradients (parameters, tokens and the kept slots' weights), over
    drawn tokens and parameters at MaxText's routing, on par with the dropless layer with the dropped
    weights zeroed from the float64 expert loop over the slots MaxText keeps (check 1), the dropless
    layer's own float64 run first held to the loop. A dropped slot's weight has no gradient."""
    arrays, cases = maxtext
    shape = cases[case]
    indices, weights = arrays[f"{case}/indices"], arrays[f"{case}/weights"]
    kept = maxtext_kept(arrays, case)
    dropless = ExpertMLP(shape["experts"], 12, 8)
    gradient = functools.partial(jax.value_and_grad, argnums=(0, 1, 2), has_aux=True)
    names = ["output", "down", "gate", "up", "tokens", "weights"]

    def run(module, values, given):
        out = jax.jit(gradient(functools.partial(objective, module, jnp.asarray(indices))))(*values, given)
        (_, output), grads = out
        return [np.asarray(leaf) for leaf in jax.tree.leaves((output, grads))]

    def held(leaves):
        """The leaves check 1 reads, the weights' gradient at the slots kept."""
        return {name: leaf[kept] if name == "weights" else leaf
                for name, leaf in zip(names, leaves, strict=True)}

    def draw(seed):
        rng = np.random.default_rng([29, seed])
        x = rng.normal(size=(shape["batch"], shape["length"], 8)).astype(np.float32)
        return nn.unbox(dropless.init(jax.random.key(seed), x, weights, indices)), x

    def reference(parameters, x):
        return held(run(dropless, (parameters, x), np.where(kept, weights, 0).astype(np.float32)))

    def widened(parameters, x):
        return jax.tree.map(lambda leaf: jnp.asarray(np.asarray(leaf), jnp.float64), (parameters, x))

    def twin(parameters, x):
        with jax.enable_x64(new_val=True):
            return held(run(dropless.clone(dtype=jnp.float64), widened(parameters, x),
                            np.where(kept, weights, 0).astype(np.float64)))

    def truth(parameters, x):
        with jax.enable_x64(new_val=True):
            def loop(values, given):
                out = expert_loop(values[1], given, indices, kept, values[0])
                return jnp.sum(jnp.sin(out)), out

            (_, output), grads = jax.value_and_grad(lambda p, xs, w: loop((p, xs), w), argnums=(0, 1, 2),
                                                    has_aux=True)(*widened(parameters, x),
                                                                  jnp.asarray(weights, jnp.float64))
            return held([np.asarray(leaf) for leaf in jax.tree.leaves((output, grads))])

    def dropping(parameters, x):
        with jax.set_mesh(spec.build(jax.devices()[:4])):
            leaves = run(dropless.clone(dispatch=dispatch, capacity_factor=shape["factor"]), (parameters, x),
                         weights)
        np.testing.assert_array_equal(np.where(kept, 0, leaves[-1]), 0)
        return held(leaves)

    # The float64 twin computes the loop's oracle: the two projections' widths, the experts a token sums
    # and the tokens a weight sums.
    first, wanted = twin(*draw(0)), truth(*draw(0))
    for name in names:
        assert_computes_the_oracle(first[name], wanted[name], f"{case} {name}",
                                   roundings=8 + 12 + shape["top_k"] + shape["batch"] * shape["length"])
    assert_as_exact(as_given(dropping), as_given(reference), draw,
                    f"{case} {dispatch} capacity drops over {spec}",
                    seeds=CAPACITY_SEEDS, truth_fn=as_given(truth))


def test_exchange_collectives_work_across_two_real_processes(tmp_path):
    assert_exchange_across_processes(Path(__file__).with_name('moe_exchange_worker.py'), tmp_path, 3e-5)


@pytest.mark.mesh(devices=4)
@pytest.mark.parametrize("dispatch", ["global", "exchange"])
def test_value_experts_drop_and_dispatch_as_their_models_routed_mlp(dispatch):
    """MoVA's value experts run under their layer's mixture (`RoutedValues.mixture`, ledger 2565): at its
    capacity factor of one, on an expert mesh under either dispatch, they compute the float64 expert loop
    over the slots MaxText's queue keeps (`capacity_positions`), each kept expert's SiLU projection at its
    router weight, the dropped slots none."""
    from dew.nn.value_experts import RoutedValues, ValueExperts

    layer = RoutedValues(ValueExperts(experts=8, top_k=2), in_features=8, features=12,
                         mixture=Mixture(experts=8, top_k=2, dispatch=dispatch, capacity_factor=1.0))
    with jax.enable_x64(new_val=True):
        x = jnp.asarray(np.random.default_rng(23).normal(size=(4, 12, 8)))
        drawn = nn.unbox(layer.init(jax.random.key(3), x))
        parameters = jax.tree.map(lambda leaf: leaf.astype(jnp.float64), drawn)
        weights, indices = layer.apply(parameters, x, method=lambda module, x: module.router(x))
        positions, capacity = capacity_positions(indices, 8, 1.0)
        kept = positions < capacity
        assert not bool(kept.all())
        kernel = parameters["params"]["v_proj"]["kernel"]
        slots = jax.nn.silu(jnp.einsum("bsi,bskio->bsko", x, kernel[indices]))
        truth = np.asarray(jnp.einsum("bsko,bsk->bso", slots, weights * kept))
        with jax.set_mesh(MeshSpec(expert=4).build(jax.devices()[:4])):
            actual = np.asarray(jax.jit(layer.apply)(parameters, x))
    assert_computes_the_oracle(actual, truth, f"value experts under {dispatch}", roundings=16)
