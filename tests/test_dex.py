"""DEX, the differential extension on chosen heads (`dew.nn.dex`, arXiv 2505.16333): its equations
against a plain float64 reference, forward and gradients, its annealed lambda, and the public path, a
loaded source with the extension attached."""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.traverse_util import flatten_dict, unflatten_dict
from reference_error import as_given, assert_as_exact

from dew.batch import FROZEN, thaw
from dew.interop import Pretrained
from dew.nn.dex import DifferentialExtension, LayerDex, annealed, diff_lambda_init, differential

LLAMA = Path(__file__).parent / "fixtures" / "hf" / "llama-tiny"
HEADS = (1, 3)
SEEDS = {"dex differential": 16, "dex through the decoder": 16}
"""Check 1's declared seeds for each site."""


def reference(attention, projection, strength, cotangent):
    """The paper's Eq. 3 and 5 in numpy, head by head (`O'_h = O_h - lambda O_h W_h` on the chosen
    heads, the others as they are), and its gradients by hand under `cotangent`: the output, then the
    gradients of `<O', cotangent>` with respect to O, W and lambda."""
    out = np.array(attention, copy=True)
    d_attention = np.array(cotangent, copy=True)
    d_projection = np.zeros_like(np.asarray(projection))
    d_strength = 0.0
    for slot, head in enumerate(HEADS):
        o, w, g = attention[:, :, head], projection[slot], cotangent[:, :, head]
        moved = o @ w
        out[:, :, head] = o - strength * moved
        d_attention[:, :, head] = g - strength * g @ w.T
        d_projection[slot] = -strength * np.einsum("bsd,bse->de", o, g)
        d_strength -= float(np.sum(moved * g))
    return out, d_attention, d_projection, np.asarray(d_strength)


def draw(seed):
    rng = np.random.default_rng([0xDE, seed])
    attention = rng.standard_normal((4, 64, 4, 16)).astype(np.float32)
    projection = (rng.standard_normal((len(HEADS), 16, 16)) / np.sqrt(16)).astype(np.float32)
    strength = np.float32(rng.uniform(0.1, 0.9))
    cotangent = rng.standard_normal(attention.shape).astype(np.float32)
    return attention, projection, strength, cotangent


@jax.jit
def dex(attention, projection, strength, cotangent):
    def inner(attention, projection, strength):
        return jnp.sum(differential(attention, HEADS, projection, strength) * cotangent)
    out = differential(attention, HEADS, projection, strength)
    return (out, *jax.grad(inner, argnums=(0, 1, 2))(attention, projection, strength))


def test_the_differential_is_the_papers_equation_forward_and_back():
    """`differential` on the chosen heads is Eq. 3 and 5 head by head, its output and its gradients with
    respect to the outputs, the projections and lambda on par with a plain numpy reference of the
    equations, from that reference in float64 (check 1)."""
    def numpy_arm(attention, projection, strength, cotangent):
        return reference(np.asarray(attention), np.asarray(projection), float(strength),
                         np.asarray(cotangent))

    def truth(attention, projection, strength, cotangent):
        wide = [np.asarray(value, np.float64) for value in (attention, projection, cotangent)]
        return reference(wide[0], wide[1], float(strength), wide[2])

    assert_as_exact(dex, as_given(numpy_arm), draw, "dex differential", seeds=SEEDS["dex differential"],
                    truth_fn=truth)


def test_a_training_step_anneals_lambda_through_the_runs_progress():
    """Within a run's step (`dew.nn.progress.stepping`) the layer reads lambda annealed over the run's
    updates: at update 0 the extended decoder computes the source's logits bit for bit whatever lambda
    has learned, and past the anneal its learned lambda, as outside a run."""
    from dew.nn.progress import Progress, stepping

    extension = DifferentialExtension(((0, HEADS),), anneal_steps=10, trains="all")
    loaded = Pretrained.load(LLAMA, dtype="float32", attention_impl="reference", dex=extension)
    source = Pretrained.load(LLAMA, dtype="float32", attention_impl="reference")
    flat = flatten_dict(loaded.variables["params"])
    for key in flat:
        if key[-1] == "dex_lambda":
            flat[key] = jnp.asarray(0.5, jnp.float32)
    variables = {**loaded.variables, "params": unflatten_dict(flat)}
    ids = jnp.asarray(np.random.default_rng(1).integers(0, 256, (2, 7)), jnp.int32)
    with stepping(Progress(jnp.asarray(0, jnp.int32))):
        start = loaded.model.apply(variables, ids)
    np.testing.assert_array_equal(np.asarray(start), np.asarray(source.model.apply(source.variables, ids)))
    with stepping(Progress(jnp.asarray(10, jnp.int32))):
        ended = loaded.model.apply(variables, ids)
    np.testing.assert_array_equal(np.asarray(ended), np.asarray(loaded.model.apply(variables, ids)))


def test_lambda_anneals_from_zero_to_its_learned_value():
    """Eq. 4: lambda(0) = 0, half way (1 - a) (t/T) lambda_init + a lambda_learn at a = 1/2, the learned
    value from T on, and the learned value outside training (no step); lambda_init by default the
    Differential Transformer's depth schedule."""
    layer = LayerDex(HEADS, 0.6, 100)
    learned = jnp.asarray(0.25, jnp.float32)
    assert float(annealed(jnp.asarray(0), layer, learned)) == 0.0
    # In float32, each operation as Eq. 4 orders it.
    a, start = np.float32(0.5), np.float32(0.5) * np.float32(0.6)
    assert float(annealed(jnp.asarray(50), layer, learned)) == (1 - a) * start + a * np.float32(0.25)
    for updates in (100, 1000):
        assert float(annealed(jnp.asarray(updates), layer, learned)) == np.float32(0.25)
    assert annealed(None, layer, learned) is learned
    chosen = DifferentialExtension(((1, (3, 1, 1)),), anneal_steps=10).layer(1)
    assert chosen == LayerDex((1, 3), diff_lambda_init(2), 10)
    assert diff_lambda_init(1) == 0.8 - 0.6
    for layers in ((), ((0, ()),), ((0, (1,)), (0, (2,)))):
        with pytest.raises(ValueError):
            DifferentialExtension(layers)


def absorbed(variables, layer: int, heads, value_dim: int):
    """A decoder's variables with layer `layer`'s DEX folded into its output projection: `o_proj(O')`
    is `o_proj` with each chosen head's block `K_h` made `(I - lambda W_h) K_h`, computed in float64,
    the extension's own parameters left out. An independent route to the extended layer's function."""
    flat = flatten_dict(variables["params"])
    prefix = next(key[:-1] for key in flat if key[-1] == "dex_lambda" and f"layers_{layer}" in key)
    strength = np.float64(flat.pop((*prefix, "dex_lambda")))
    projection = np.asarray(flat.pop((*prefix, "dex_projection")), np.float64)
    kernel = np.array(flat[(*prefix, "o_proj", "kernel")], np.float64)
    for slot, head in enumerate(heads):
        rows = slice(head * value_dim, (head + 1) * value_dim)
        kernel[rows] = (np.eye(value_dim) - strength * projection[slot]) @ kernel[rows]
    flat[(*prefix, "o_proj", "kernel")] = kernel.astype(np.float32)
    return {**variables, "params": unflatten_dict(flat)}


def test_a_loaded_source_takes_dex_and_computes_its_folded_output_projection():
    """Through the public path: a source loaded with DEX on layer 0's heads 1 and 3 adds their
    projections and a zero lambda alone, and computes the source's logits bit for bit until they train;
    with lambda and the projections moved, its logits are the source's with layer 0's output projection
    folded through the extension (`absorbed`), on par from the extended decoder's float64 (check 1)."""
    extension = DifferentialExtension(((0, HEADS),), anneal_steps=10)
    source = Pretrained.load(LLAMA, dtype="float32", attention_impl="reference")
    recipe = Pretrained.load(LLAMA, dtype="float32", attention_impl="reference", dex=extension)
    # The paper's recipe: the extension and the attention's key, value and output projections train, the
    # rest frozen.
    trained = {key[-2] if key[-1] in ("kernel", "bias") else key[-1]
               for key in flatten_dict(recipe.variables["params"])}
    assert trained == {"k_proj", "v_proj", "o_proj", "dex_projection", "dex_lambda"}
    assert FROZEN in recipe.variables
    loaded = Pretrained.load(LLAMA, dtype="float32", attention_impl="reference",
                             dex=replace(extension, trains="all"))
    assert thaw(recipe.variables).keys() == loaded.variables.keys()
    added = {key[-1] for key in flatten_dict(loaded.variables["params"])} - {
        key[-1] for key in flatten_dict(source.variables["params"])}
    assert added == {"dex_projection", "dex_lambda"}
    ids = jnp.asarray(np.random.default_rng(0).integers(0, 256, (2, 7)), jnp.int32)
    np.testing.assert_array_equal(np.asarray(loaded.model.apply(loaded.variables, ids)),
                                  np.asarray(source.model.apply(source.variables, ids)))
    value_dim = loaded.model.features_per_head

    def moved(seed):
        rng = np.random.default_rng([0xDEC, seed])
        flat = flatten_dict(loaded.variables["params"])
        for key in flat:
            if key[-1] == "dex_lambda":
                flat[key] = jnp.asarray(rng.uniform(0.2, 0.8), jnp.float32)
            elif key[-1] == "dex_projection":
                drawn = rng.standard_normal(flat[key].shape) / np.sqrt(value_dim)
                flat[key] = jnp.asarray(drawn, jnp.float32)
        return {**loaded.variables, "params": unflatten_dict(flat)}, jnp.asarray(rng.integers(0, 256, (2, 7)))

    run = jax.jit(loaded.model.apply)
    plain = jax.jit(source.model.apply)
    wide = jax.jit(loaded.model.clone(dtype=jnp.float64).apply)
    def folded(variables, ids):
        return plain(absorbed(variables, 0, HEADS, value_dim), ids)

    assert_as_exact(run, as_given(folded), moved, "dex through the decoder",
                    seeds=SEEDS["dex through the decoder"], truth_fn=wide)


def test_a_run_names_dex_on_the_command_line_and_its_record_rebuilds_it():
    """`--source.dex` takes the extension's record as JSON, every run kind's (`SourceOptions.dex`), which its
    source's load reads (`RunConfig.source_keywords`) and its run record writes and reads back; the
    decoder that carries it records it beside its fields and rebuilds from that record."""
    from dew.config.lm import LMRunConfig
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.registry import from_record, to_record

    extension = DifferentialExtension(((0, (1, 3)), (1, (0,))), anneal_steps=200)
    run = LMRunConfig.cli(["--source.dex", '{"layers": [[0, [1, 3]], [1, [0]]], "anneal_steps": 200}',
                           "--trainer.multi-host", "False"])
    assert run.source.dex == extension
    assert run.source_keywords(frozen=False)["dex"] == extension
    assert LMRunConfig.from_dict(run.to_dict()).source == run.source
    loaded = Pretrained.load(LLAMA, dtype="float32", attention_impl="reference",
                             **run.source.keywords())
    model = loaded.model
    assert isinstance(model, CausalTransformer) and model.dex == extension
    assert from_record(CausalTransformer, to_record(model, CausalTransformer)).dex == extension
