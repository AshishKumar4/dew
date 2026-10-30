"""DEX, the differential extension on chosen heads (`dew.nn.dex`, arXiv 2505.16333): its equations
against a plain float64 reference, forward and gradients, its annealed lambda, and the public path, a
loaded source with the extension attached."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.traverse_util import flatten_dict, unflatten_dict
from model_support import TINY_DECODER
from reference_error import as_given, assert_as_exact, widened

import dew.nn.dex
from dew.batch import FROZEN, thaw
from dew.interop import Pretrained, PretrainedDecoder
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.dex import (
    DEX_COLLECTION,
    DexStart,
    DifferentialExtension,
    LayerDex,
    annealed,
    diff_lambda_init,
    differential,
)
from dew.nn.progress import END, Progress

LLAMA = Path(__file__).parent / "fixtures" / "hf" / "llama-tiny"
HEADS = (1, 3)
SEEDS = {"dex differential": 256, "dex through the decoder": 8, "dex at the anneal's start": 8,
         "dex gradients through the decoder": 256, "dex scanned": 8, "dex pipelined": 8,
         "dex exported in float32": 8, "dex exported in bfloat16": 8}
"""Check 1's declared seeds for each site, sized paired by `seeds_for` on armada (IGDEXS, IGDEXS4)."""


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
    layer, begins = LayerDex(len(HEADS), 100), jnp.asarray(0.6, jnp.float32)
    learned = jnp.asarray(0.25, jnp.float32)
    assert float(annealed(jnp.asarray(0), layer, begins, learned)) == 0.0
    # In float32, each operation as Eq. 4 orders it.
    a, start = np.float32(0.5), np.float32(0.5) * np.float32(0.6)
    assert float(annealed(jnp.asarray(50), layer, begins, learned)) == (1 - a) * start + a * np.float32(0.25)
    for updates in (100, 1000):
        assert float(annealed(jnp.asarray(updates), layer, begins, learned)) == np.float32(0.25)
    assert annealed(None, layer, begins, learned) is learned
    extension = DifferentialExtension(((1, (3, 1, 1)),), anneal_steps=10)
    assert extension.layer(1) == LayerDex(2, 10)
    assert extension.start(1) == DexStart((1, 3), diff_lambda_init(2))
    assert extension.layer(0) is None and extension.start(0) is None
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


def traced_wide(tree):
    """`tree`'s floating leaves in float64 inside a traced function, where `widened`'s numpy cannot read
    them."""
    return jax.tree.map(lambda leaf: leaf.astype(jnp.float64) if jnp.issubdtype(leaf.dtype, jnp.floating)
                        else leaf, tree)


def loads(trains: str = "all", dtype: str = "float32"):
    """The source and the source with DEX on layer 0's heads 1 and 3 (`HEADS`), each loaded computing
    and storing in `dtype`."""
    extension = DifferentialExtension(((0, HEADS),), anneal_steps=10, trains=trains)
    return (Pretrained.load(LLAMA, dtype=dtype, param_dtype=dtype, attention_impl="reference"),
            Pretrained.load(LLAMA, dtype=dtype, param_dtype=dtype, attention_impl="reference", dex=extension))


def moved_draw(variables, value_dim: int):
    """A draw of the extended decoder's variables with lambda and the projections moved off their start,
    beside a row of tokens and a cotangent on its logits."""
    def draw(seed):
        rng = np.random.default_rng([0xDEC, seed])
        flat = flatten_dict(variables["params"])
        for key in flat:
            if key[-1] == "dex_lambda":
                flat[key] = jnp.asarray(rng.uniform(0.2, 0.8), flat[key].dtype)
            elif key[-1] == "dex_projection":
                drawn = rng.standard_normal(flat[key].shape) / np.sqrt(value_dim)
                flat[key] = jnp.asarray(drawn, flat[key].dtype)
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
    held = flatten_dict(loaded.variables[DEX_COLLECTION])
    np.testing.assert_array_equal(held[("layers_0", "self_attn", "dex_heads")], HEADS)
    assert float(held[("layers_0", "self_attn", "dex_lambda_init")]) == np.float32(diff_lambda_init(1))
    # A run's checkpoint keeps the extension, and reads back as a bundle of the decoder it extends.
    rebuilt = PretrainedDecoder.from_model(loaded.model, loaded.variables)
    assert rebuilt.model.dex == loaded.model.dex


def exported_site(directory: Path, dtype: str):
    """Check 1's arms for an export of the extended decoder computing and storing in `dtype`: the
    source loaded back from what `save` writes of a draw's variables, the extended decoder's own
    unfused forward outside a run (its learned lambda), a draw of moved variables, and that forward's
    float64 as truth."""
    _, loaded = loads(dtype=dtype)

    @as_given
    def reloaded(variables, ids, _):
        loaded.save(directory, variables=variables)
        back = Pretrained.load(directory, dtype=dtype, param_dtype=dtype, attention_impl="reference")
        return np.asarray(back.model.apply(back.variables, ids), np.float32)

    wide = loaded.model.clone(dtype=jnp.float64)
    live = jax.jit(lambda variables, ids, _: loaded.model.apply(variables, ids).astype(jnp.float32))
    return (reloaded, live,
            moved_draw(loaded.variables, loaded.model.features_per_head),
            jax.jit(lambda variables, ids, _: wide.apply(traced_wide(variables), ids)))


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_an_export_writes_dex_folded_into_its_output_projections(tmp_path, dtype):
    """The source's format holds no extension, so an export writes each layer's differential at its
    learned lambda folded into its output projection (ledger 2478): the source loaded back from it
    computes what the extended decoder's own unfused forward does outside a run, on par with it from
    its float64 (check 1), in float32 and computing and storing in bf16, where the fold rounds where the
    unfused layer does not. A layer whose output gate scales its heads' outputs before the projection
    does not fold, and its export refuses."""
    reloaded, live, draw, truth = exported_site(tmp_path, dtype)
    assert_as_exact(reloaded, live, draw, f"dex exported in {dtype}", seeds=SEEDS[f"dex exported in {dtype}"],
                    truth_fn=truth)
    gated = every_layer(output_gate=True)
    with pytest.raises(ValueError):
        gated.dex_exported(gated.init(jax.random.key(0), jnp.zeros((1, 4), jnp.int32)))


def test_an_export_folds_the_operands_a_layer_multiplies_in():
    """A layer computing in bf16 multiplies its chosen heads by lambda and `W_h` rounded to bf16
    (`differential`), so its export folds those (`dew.nn.dex.folded`): `W = (1 + 2^-9) I` at lambda 1
    rounds to `I` and folds the head's rows of `o_proj` to zero, where the float32 masters would leave
    `-2^-9 K_h` (review of ledger 2478)."""
    width, heads = 8, 2
    kernel = np.arange(1, heads * width * 4 + 1, dtype=np.float32).reshape(heads * width, 4)
    layer = {"o_proj": {"kernel": kernel}, "dex_lambda": np.float32(1.0),
             "dex_projection": np.eye(width, dtype=np.float32)[None] * np.float32(1 + 2 ** -9)}
    variables = {"params": {"self_attn": layer},
                 DEX_COLLECTION: {"self_attn": {"dex_heads": np.asarray([1], np.int32),
                                                "dex_lambda_init": np.float32(0.2)}}}
    rows = slice(width, 2 * width)
    in_bf16 = dew.nn.dex.folded(variables, jnp.bfloat16)["params"]["self_attn"]["o_proj"]["kernel"]
    np.testing.assert_array_equal(in_bf16[rows], np.zeros((width, 4), np.float32))
    np.testing.assert_array_equal(in_bf16[:width], kernel[:width])
    in_float32 = dew.nn.dex.folded(variables, None)["params"]["self_attn"]["o_proj"]["kernel"]
    np.testing.assert_array_equal(in_float32[rows], (-(2 ** -9) * kernel[rows]).astype(np.float32))


def test_a_run_starts_from_its_source_and_its_learned_lambda_once_annealed():
    """Passed a run's step's progress (`dew.nn.progress`), the layer reads lambda annealed over the run's
    updates: at update 0 the extended decoder computes its source's logits whatever its extension holds,
    and from the anneal's end on, as outside a run, the source's forward over the folded output
    projection (`folded`), each on par from the float64 of what it is held to (check 1)."""
    source, loaded = loads()
    draw = moved_draw(loaded.variables, loaded.model.features_per_head)

    def stepped(updates: int | None):
        def run(variables, ids, _):
            progress = END if updates is None else Progress(jnp.asarray(updates, jnp.int32))
            return loaded.model.apply(variables, ids, progress=progress)
        return jax.jit(run)

    plain = jax.jit(lambda variables, ids, _: source.model.apply(source.variables, ids))
    wide = source.model.clone(dtype=jnp.float64)
    assert_as_exact(stepped(0), plain, draw, "dex at the anneal's start",
                    seeds=SEEDS["dex at the anneal's start"],
                    truth_fn=jax.jit(lambda variables, ids, _: wide.apply(widened(source.variables), ids)))
    through = jax.jit(lambda variables, ids, _: source.model.apply(folded(variables), ids))
    wide_through = jax.jit(lambda variables, ids, _: wide.apply(folded(traced_wide(variables)), ids))
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
                variables, cotangent = traced_wide(variables), traced_wide(cotangent)

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


EVERY_LAYER = DifferentialExtension(((0, (0, 1)), (1, (1, 3)), (2, (2, 0)), (3, (3, 2))), anneal_steps=10)
"""DEX on each of four layers, each choosing two heads of its own, lambda_init the depth schedule's."""


def every_layer(**fields) -> CausalTransformer:
    return CausalTransformer(**{"vocab_size": 64, **TINY_DECODER, "num_layers": 4, "dtype": jnp.float32,
                                "attention_impl": "reference", "dex": EVERY_LAYER, **fields})


def mid_anneal(model):
    """`model`'s logits at a run's step half way through the anneal (`dew.nn.progress`), compiled."""
    @jax.jit
    def run(variables, tokens):
        return model.apply(variables, tokens, progress=Progress(jnp.asarray(5, jnp.int32)))
    return run


def scanned_site():
    """Check 1's arms for the scanned run of `every_layer`: the scanned and plain decoders mid-anneal, a
    draw of their variables with each layer's lambda and projections moved off their start, and the
    plain decoder's float64 as truth."""
    plain = every_layer()
    start = plain.init(jax.random.key(0), jnp.zeros((2, 7), jnp.int32))

    def draw(seed):
        rng = np.random.default_rng([0xD5, seed])
        flat = flatten_dict(start["params"])
        for key in flat:
            if key[-1] in ("dex_lambda", "dex_projection"):
                flat[key] = jnp.asarray(rng.uniform(-0.5, 0.5, flat[key].shape), jnp.float32)
        tokens = jnp.asarray(rng.integers(0, 64, (2, 7)), jnp.int32)
        return {**start, "params": unflatten_dict(flat)}, tokens

    wide = mid_anneal(plain.clone(dtype=jnp.float64))
    return (mid_anneal(every_layer(scan_layers=True)), mid_anneal(plain), draw,
            jax.jit(lambda variables, tokens: wide(traced_wide(variables), tokens)))


def test_layers_that_choose_their_own_heads_scan_and_pipeline_as_one_run():
    """Each layer's chosen heads and lambda_init are its own variables, so layers that choose as many
    heads share one spec (ledger 2476): the four scan as one run and split over two pipeline stages,
    and the scanned run mid-anneal, each layer at its own heads and lambda_init, is on par with the
    plain loop from its float64 (check 1)."""
    assert every_layer().stage_split(2).per_stage == 2
    scanned, plain, draw, truth = scanned_site()
    variables, _ = draw(0)
    assert every_layer(scan_layers=True).apply(variables, method=lambda module: module.groups) == ((0, 4),)
    assert_as_exact(scanned, plain, draw, "dex scanned", seeds=SEEDS["dex scanned"], truth_fn=truth)


@pytest.mark.mesh
def test_a_pipelined_run_reads_its_rows_progress():
    """A run's progress rides the call's metadata one row's apiece (`AttentionMetadata.progress`), which
    a pipeline's microbatches slice as they slice its other row inputs (review of ledger 2537): the four
    layers over two stages of two microbatches each, mid-anneal, are on par with the plain loop from its
    float64 (check 1), where a scalar progress could not be cut into microbatches."""
    from dew.distributed.batches import shard_batch
    from dew.distributed.placement import Layout, MeshSpec
    from dew.nn.sharding import pipeline_microbatches

    _, plain, draw, truth = scanned_site()
    mesh = MeshSpec(fsdp=2, stage=2).build()

    def rows(seed):
        variables, tokens = draw(seed)
        return variables, jnp.tile(tokens, (4, 1))

    @as_given
    def staged(variables, tokens):
        with jax.set_mesh(mesh), pipeline_microbatches(2):
            laid = jax.device_put(variables, Layout(min_shard=16).shardings(mesh, variables))
            return jax.device_get(mid_anneal(every_layer())(laid, shard_batch(mesh, tokens)))

    assert_as_exact(staged, plain, rows, "dex pipelined", seeds=SEEDS["dex pipelined"], truth_fn=truth)


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
