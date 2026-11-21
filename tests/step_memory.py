"""The ordering rule's guard (rows 6027, 6471): the step users run, `dew.training.Trainer`'s, over a decoder
of every registered mixer (`dew.registry.mixers`) and of attention blocks holding each
feature's parameters (`FEATURES`), its figures read from the programs' own trees (`step_bytes`) and its bound
asserted by `holds`. Its cases (`cases`) are shared out among files of their own (tests/test_step_memory_*.py,
`shard`), so none outruns one task's deadline and a mixer the registry gains is guarded."""

import math

import jax
import jax.numpy as jnp
from flax import linen as nn

from dew.distributed.placement import Layout
from dew.mesh import DEFAULT_RULES
from dew.nn.kernels import generation

MIXER_ROWS, MIXER_LENGTH, MIXER_WIDTH, MIXER_HEAD = 2, 2048, 512, 64


STEP_LAYOUT = Layout(min_shard=MIXER_WIDTH)
"""The trainer's layout for the guard's decoders: its parameters split over the data axis wherever a dimension
is `MIXER_WIDTH` or more, as a run's are."""


def step_bytes(kind: str, layers: int, feature: str = "plain", copies: bool = True) -> dict[str, int]:
    """A device's bytes in the step users run (`dew.training.Trainer`, compiled ahead on two compile-only
    H100s splitting its rows, AdamW's update after the backward), of a `layers`-deep decoder of the
    registered mixer `kind`, its blocks of `feature` (`FEATURES`), every block rematerialized whole, in bf16
    over fp32 parameters, 512 wide over rows of 2,048 tokens, with the trainer's narrow copies or without
    them (`chosen(narrow_copies='off')`): what its buffers hold beside its arguments at its peak, as XLA's
    heap simulation states it (`AheadOfTime.live_bytes`), its temporaries' one allocation (that, with the
    room their packing takes), its arguments and outputs; what its forward saves for its reverse
    (`saved_residuals`, its arguments aside: each block's input, the residual stream and any state a block
    passes the next), a device's share of its rows; and its gradients' bytes as the update reads them, each
    parameter's shard where the layout places it (`Layout.shardings`) in the dtype its gradient takes (its
    narrow copy's, `narrowed_paths`, or its own), which a step whose update runs after its backward holds
    until the update reads them. Its fields beyond the defaults are test_mixers' (`KIND_FIELDS`), but its
    heads' width the model's (`MIXER_HEAD`), a QSA indexer's turned by the layer's rotary."""
    from jax._src.ad_checkpoint import saved_residuals
    from test_mixers import KIND_FIELDS
    from token_support import token_rows

    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.backbones.decoder_block import REMAT_POLICIES
    from dew.nn.kernels.generation import compile_only_gpus
    from dew.objectives.lm import LMObjective
    from dew.optim import OptimConfig
    from dew.registry import mixers
    from dew.training import MeshSpec, Trainer
    from dew.training.ahead import compile_ahead
    from dew.training.narrow import program_reads

    devices = 2
    fields, overrides = KIND_FIELDS.get(kind, ({}, {}))
    fields = {**fields, **({"indexer_head_dim": MIXER_HEAD} if "indexer_head_dim" in fields else {})}
    model = CausalTransformer(**{"vocab_size": 512, "emb_features": MIXER_WIDTH, "num_layers": layers,
                                 "num_heads": 8, "head_dim": MIXER_HEAD, "mlp_features": 4 * MIXER_WIDTH,
                                 "max_seq_len": MIXER_LENGTH, "dtype": "bfloat16",
                                 "remat": REMAT_POLICIES["full"], "mixer": mixers.build(kind, **fields),
                                 **overrides, **features()[feature]})
    trainer = Trainer(LMObjective(model, MIXER_LENGTH),
                      OptimConfig(optimizer="adamw", learning_rate=1e-3).build(None), key=jax.random.key(0),
                      mesh=MeshSpec().build(compile_only_gpus("sm90", devices, 2**36)), layout=STEP_LAYOUT,
                      split=1)
    with generation.chosen(**({} if copies else {"narrow_copies": "off"})):
        ahead = compile_ahead(trainer, token_rows(512, MIXER_ROWS, MIXER_LENGTH + 1))
    tokens = jnp.zeros((MIXER_ROWS, MIXER_LENGTH), jnp.int32)
    # The model's tree as the trainer's start drew it (`Objective.drawn_shapes`, recorded by its trace), not
    # traced again (13 s of a 4-block sparse decoder's case, rows 6766 on).
    variables = trainer.objective.model_variables(trainer.objective.drawn_shapes())

    def loss(params, constants, tokens):
        return jnp.sum(jnp.asarray(model.apply({"params": params, **constants}, tokens), jnp.float32))

    constants = {key: value for key, value in variables.items() if key != "params"}
    with generation.assumed("sm90"), nn.logical_axis_rules(DEFAULT_RULES):
        # The model traced once, its program replayed for both reads: what its backward saves and what
        # it casts narrower (a decoder's trace is seconds of a case, rows 6766 on).
        closed = jax.make_jaxpr(loss)(variables["params"], constants, tokens)

        def replayed(params, constants, tokens):
            return jax.core.eval_jaxpr(closed.jaxpr, closed.consts,
                                       *jax.tree.leaves((params, constants, tokens)))[0]

        saved = sum(aval.size * aval.dtype.itemsize
                    for aval, source in saved_residuals(replayed, variables["params"], constants, tokens)
                    if not source.startswith("from the argument"))
        paths = program_reads(closed, variables["params"]).casts if copies else {}
    placement = STEP_LAYOUT.shardings(trainer.device_mesh, {"params": variables["params"]})["params"]
    leaves = jax.tree_util.tree_leaves_with_path(variables["params"])
    gradients = sum(math.prod(sharding.shard_shape(leaf.shape))
                    * jnp.dtype(paths.get(tuple(key.key for key in path), leaf.dtype)).itemsize
                    for (path, leaf), sharding in zip(leaves, jax.tree.leaves(placement), strict=True))
    return {"live": ahead.live_bytes - ahead.argument_bytes, "temp": ahead.temp_bytes,
            "argument": ahead.argument_bytes, "output": ahead.output_bytes, "alias": ahead.alias_bytes,
            "saved": saved // devices, "gradients": gradients}


SHARDS = 5
"""The files the cases are shared out among (tests/test_step_memory_<i>.py)."""

FEATURES = ("plain", "sinks", "biases", "ngpt", "experts")
"""The parameters a block holds beside its mixer's and its MLP's, each through a custom reverse mode of its
own: none, an attention's sinks (gpt-oss's: `attention_sinks`), its projections' biases stored fp32 (the
once-rounded biased product's tangent, `dense._widened`), nGPT's rates and gains (`ngpt`) and a mixture of
experts' expert projections (`moe.expert_projection`)."""


def features() -> dict[str, dict]:
    """Each of `FEATURES`' decoder fields."""
    from dew.nn.backbones.decoder_block import Mixture
    from dew.nn.ngpt import NGPT

    return {"plain": {}, "sinks": {"attention_sinks": True},
            "biases": {"attention_bias": True, "mlp_bias": True},
            "ngpt": {"ngpt": NGPT(std=0.02, alpha=0.05), "pre_norms": False, "qk_norm": False},
            "experts": {"mixture": Mixture(experts=4, top_k=2)}}


def cases() -> list[tuple[str, str]]:
    """Every registered mixer's plain blocks, and the attention mixer's blocks of each of `FEATURES`'
    parameters."""
    from dew.registry import mixers

    return ([(kind, "plain") for kind in sorted(mixers.paths)]
            + [("attention", feature) for feature in FEATURES[1:]])


def shard(index: int) -> list[tuple[str, str]]:
    """`cases()` the `index`th file of `SHARDS` holds: every `SHARDS`th from its own."""
    return cases()[index::SHARDS]


def holds(kind: str, feature: str, copies: bool) -> None:
    """The step users run (`dew.training.Trainer`, its update after its backward) over a decoder of the
    registered mixer `kind` in blocks holding `feature`'s parameters (`FEATURES`), every block rematerialized,
    on two compile-only H100s splitting its rows, with the trainer's narrow copies or without them (`copies`):
    what a device's buffers hold beside its arguments at the step's peak (XLA's heap simulation, apart from
    the room its temporaries' packing takes, which its message states beside) grows from two blocks to four by
    no more than what the forward saves (each block's input, as its rematerialization keeps it: the residual
    stream and any state a block passes the next, `saved_residuals`) and the added blocks' gradients in the
    dtype the update reads them, which it holds until the update runs, both read from the programs' own trees
    (`step_bytes`). Each reverse mode that forms a parameter gradient returns its input's cotangent only with
    it formed (`derivatives.with_gradients_formed`, or its dual for a transposed tangent rule), so XLA puts no
    block's cotangents off past their layer (rows 6027, 6173, 6471); a mixer or a block's parameter whose
    reverse mode holds more fails here, its temporaries', arguments' and outputs' growth stated beside."""
    two, four = (step_bytes(kind, layers, feature, copies) for layers in (2, 4))
    grown = {key: four[key] - two[key] for key in two}
    assert grown["live"] <= grown["saved"] + grown["gradients"], f"{grown}"
