"""DEX, the differential extension on chosen heads (`dew.nn.dex`, arXiv 2505.16333): its equations
against a plain float64 reference, forward and gradients, its annealed lambda, and the public path, a
loaded source with the extension attached."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.traverse_util import flatten_dict, unflatten_dict
from reference_error import as_given, assert_as_exact, widened

from dew.batch import FROZEN, thaw
from dew.interop import Pretrained, PretrainedDecoder
from dew.nn.dex import DifferentialExtension, LayerDex, annealed, diff_lambda_init, differential

LLAMA = Path(__file__).parent / "fixtures" / "hf" / "llama-tiny"
HEADS = (1, 3)
SEEDS = {"dex differential": 256, "dex through the decoder": 8, "dex at the anneal's start": 8,
         "dex gradients through the decoder": 8}
"""Check 1's declared seeds for each site, sized paired by `seeds_for` on armada (IGDEXS, IGDEXS2)."""


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


def folded(variables):
    """A decoder's variables with each layer's DEX folded into its output projection: `o_proj(O')` is
    `o_proj` with each chosen head's block `K_h` made `K_h - lambda W_h K_h`, `(I - lambda W_h) K_h`, the
    extension's own parameters left out, in jnp so it differentiates: an independent route to the
    extended layer's function, through the source decoder's own forward (Eq. 3 and 5)."""
    flat = flatten_dict(variables["params"])
    for prefix in [key[:-1] for key in flat if key[-1] == "dex_lambda"]:
        strength = flat.pop((*prefix, "dex_lambda"))
        projection = flat.pop((*prefix, "dex_projection"))
        kernel = flat[(*prefix, "o_proj", "kernel")]
        width = projection.shape[-1]
        for slot, head in enumerate(HEADS):
            rows = slice(head * width, (head + 1) * width)
            block = kernel[rows]
            moved = strength.astype(block.dtype) * projection[slot].astype(block.dtype) @ block
            kernel = kernel.at[rows].set(block - moved)
        flat[(*prefix, "o_proj", "kernel")] = kernel
    return {**variables, "params": unflatten_dict(flat)}


def loads(trains: str = "all"):
    """The source and the source with DEX on layer 0's heads 1 and 3 (`HEADS`), each loaded in float32."""
    extension = DifferentialExtension(((0, HEADS),), anneal_steps=10, trains=trains)
    return (Pretrained.load(LLAMA, dtype="float32", attention_impl="reference"),
            Pretrained.load(LLAMA, dtype="float32", attention_impl="reference", dex=extension))


def moved_draw(variables, value_dim: int):
    """A draw of the extended decoder's variables with lambda and the projections moved off their start,
    beside a row of tokens and a cotangent on its logits."""
    def draw(seed):
        rng = np.random.default_rng([0xDEC, seed])
        flat = flatten_dict(variables["params"])
        for key in flat:
            if key[-1] == "dex_lambda":
                flat[key] = jnp.asarray(rng.uniform(0.2, 0.8), jnp.float32)
            elif key[-1] == "dex_projection":
                drawn = rng.standard_normal(flat[key].shape) / np.sqrt(value_dim)
                flat[key] = jnp.asarray(drawn, jnp.float32)
        ids = jnp.asarray(rng.integers(0, 256, (2, 7)), jnp.int32)
        cotangent = jnp.asarray(rng.standard_normal((2, 7, 256)), jnp.float32)
        return {**variables, "params": unflatten_dict(flat)}, ids, cotangent
    return draw


def test_a_loaded_source_takes_dex_by_the_papers_recipe():
    """A source loaded with DEX adds its chosen heads' projections and a lambda alone, and by the
    paper's recipe trains them and every softmax attention layer's key, value and output projections,
    the rest frozen; with every parameter training, the same tree unsplit."""
    source, recipe = loads("attention")
    trained = {key[-2] if key[-1] in ("kernel", "bias") else key[-1]
               for key in flatten_dict(recipe.variables["params"])}
    assert trained == {"k_proj", "v_proj", "o_proj", "dex_projection", "dex_lambda"}
    assert FROZEN in recipe.variables
    _, loaded = loads()
    assert thaw(recipe.variables).keys() == loaded.variables.keys()
    added = {key[-1] for key in flatten_dict(loaded.variables["params"])} - {
        key[-1] for key in flatten_dict(source.variables["params"])}
    assert added == {"dex_projection", "dex_lambda"}
    # The source's format holds no extension: its export refuses it, where a run's checkpoint, which keeps
    # it, reads back as a bundle of the decoder it extends.
    with pytest.raises(ValueError):
        loaded.export()
    rebuilt = PretrainedDecoder.from_model(loaded.model, loaded.variables)
    assert rebuilt.model.dex == loaded.model.dex


def test_a_run_starts_from_its_source_and_its_learned_lambda_once_annealed():
    """Within a run's step (`dew.nn.progress.stepping`) the layer reads lambda annealed over the run's
    updates: at update 0 the extended decoder computes its source's logits whatever its extension holds,
    and from the anneal's end on, as outside a run, the source's forward over the folded output
    projection (`folded`), each on par from the float64 of what it is held to (check 1)."""
    from dew.nn.progress import Progress, stepping

    source, loaded = loads()
    draw = moved_draw(loaded.variables, loaded.model.features_per_head)

    def stepped(updates: int | None):
        def run(variables, ids, _):
            if updates is None:
                return loaded.model.apply(variables, ids)
            with stepping(Progress(jnp.asarray(updates, jnp.int32))):
                return loaded.model.apply(variables, ids)
        return jax.jit(run)

    plain = jax.jit(lambda variables, ids, _: source.model.apply(source.variables, ids))
    wide = source.model.clone(dtype=jnp.float64)
    assert_as_exact(stepped(0), plain, draw, "dex at the anneal's start",
                    seeds=SEEDS["dex at the anneal's start"],
                    truth_fn=jax.jit(lambda variables, ids, _: wide.apply(widened(source.variables), ids)))
    through = jax.jit(lambda variables, ids, _: source.model.apply(folded(variables), ids))
    wide_through = jax.jit(lambda variables, ids, _: wide.apply(folded(widened(variables)), ids))
    for updates in (10, None):
        assert_as_exact(stepped(updates), through, draw, "dex through the decoder",
                        seeds=SEEDS["dex through the decoder"], truth_fn=wide_through)


def test_dex_trains_through_the_decoder_as_its_folded_output_projection():
    """Gradients through the loaded decoder: of the logits under a cotangent, with respect to DEX's
    projections and lambda and layer 0's key, value and output projections, on par with the same
    gradients through the source's forward over the folded output projection (`folded`), from the
    folded route in float64 (check 1); every DEX gradient nonzero, so an update moves the extension."""
    source, loaded = loads()
    draw = moved_draw(loaded.variables, loaded.model.features_per_head)

    def kept(gradients) -> dict:
        flat = flatten_dict(gradients)
        return {"/".join(key): leaf for key, leaf in flat.items()
                if "layers_0" in key and (key[-1] in ("dex_lambda", "dex_projection")
                                          or key[-2] in ("k_proj", "v_proj", "o_proj"))}

    def gradients(forward, wide: bool = False):
        @jax.jit
        def run(variables, ids, cotangent):
            # The float64 route differentiates its widened weights, its gradients float64 too.
            if wide:
                variables, cotangent = widened(variables), widened(cotangent)

            def loss(params):
                return jnp.sum(forward({**variables, "params": params}, ids) * cotangent)
            return kept(jax.grad(loss)(variables["params"]))
        return run

    extended = gradients(loaded.model.apply)
    through = gradients(lambda variables, ids: source.model.apply(folded(variables), ids))
    wide = source.model.clone(dtype=jnp.float64)
    wide_through = gradients(lambda variables, ids: wide.apply(folded(variables), ids), wide=True)
    assert_as_exact(extended, through, draw, "dex gradients through the decoder",
                    seeds=SEEDS["dex gradients through the decoder"], truth_fn=wide_through)
    found = extended(*draw(0))
    assert all(np.any(np.asarray(leaf) != 0) for name, leaf in found.items() if "dex_" in name)


def test_a_run_names_dex_on_the_command_line_and_its_record_rebuilds_it():
    """`--source.dex` takes the extension's record as JSON, every run kind's (`SourceOptions.dex`), which
    its source's load reads (`RunConfig.source_keywords`) and its run record writes and reads back; the
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
