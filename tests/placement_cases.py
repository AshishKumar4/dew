"""The models Dew builds, each case a module and the inputs it initializes on, whose placement
tests/test_placement.py holds to the layout's typed invariants (T69: rows 1153 and 3453; ledger 4423).

The cases are the catalogue the precision and protocol tests build from (`tiny_models`: every registered
architecture and the decoder's mixtures and fixtures' mixers), a decoder over each registered mixer
(`test_mixers`'s fields), and declaring modules in forms no model there reaches: a flat and a headed
`NormalAttention`, and `GatedMLP` packed, ungated and biased."""

import functools

import jax
import jax.numpy as jnp
from test_mixers import KIND_FIELDS, tiny
from tiny_models import DECODERS, build_model, tiny_inputs

from dew.batch import unboxed
from dew.distributed.placement import Layout, MeshSpec
from dew.nn.attention_blocks import NormalAttention
from dew.nn.mlp import GatedMLP
from dew.nn.sharding import boxed, boxed_axes, declared_axes, ruled_boxes
from dew.registry import mixers, models

LAYOUT = MeshSpec(fsdp=2, expert=2, tensor=2)
"""The layout every case is placed over, on eight devices: each parameter axis a declaration names has a
mesh axis of two to land on."""

MIN_SHARD = 16
"""The fewest elements a leaf shards at, so a tiny model's declarations decide its placement rather than
its size."""

def _components() -> dict:
    """The declaring modules no catalogued model reaches in each of their forms, each with its inputs."""
    tokens = jnp.zeros((2, 5, 32))
    return {
        "NormalAttention+flat": lambda: (NormalAttention(32, heads=2, dim_head=8, flat=True,
                                                         attention_impl="reference"), (tokens,), {}),
        "NormalAttention+headed": lambda: (
            NormalAttention(32, heads=2, dim_head=8, attention_impl="reference"), (tokens,), {}),
        "GatedMLP+packed": lambda: (GatedMLP(16, 8, activation="geglu", packed=True,
                                             packed_order=("up_proj", "gate_proj")),
                                    (jnp.zeros((1, 8)),), {}),
        "GatedMLP+ungated": lambda: (GatedMLP(16, 8, activation="gelu"), (jnp.zeros((1, 8)),), {}),
        "GatedMLP+bias": lambda: (GatedMLP(16, 8, use_bias=True), (jnp.zeros((1, 8)),), {}),
    }


def _features() -> dict:
    """Decoders that turn on a feature whose modules declare axes no catalogued model reaches: continual
    learning's levels, DEX's chosen heads, routed value experts, Engram's hashed tables, nGPT's step
    rates (Dory's fixture), learned positions beside a head bias, an attention gated by its own
    projection, Qwen3-Next's fused delta-net projections and a shared expert's gate."""
    from dew.nn.backbones.decoder_block import Mixture
    from dew.nn.continuum import ContinuumLevel
    from dew.nn.dex import DifferentialExtension
    from dew.nn.engram import Engram
    from dew.nn.hyper_connections import HyperConnections
    from dew.nn.value_experts import ValueExperts
    from dew.registry import mixers

    delta = {"class": "gated_delta_net", "fields": {
        "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 8,
        "linear_value_head_dim": 8, "linear_conv_kernel_dim": 4, "fused_in_proj": True}}

    ids = (jnp.zeros((1, 8), jnp.int32),)
    fields = {
        "continuum": {"continuum": ContinuumLevel(hidden=4)},
        "dex": {"dex": DifferentialExtension(((0, (0, 1)), (1, (1, 3))), anneal_steps=10)},
        # Value experts give the values of the layers a mixture routes.
        "value_experts": {"value_experts": ValueExperts(experts=4, top_k=2),
                          "mixture": Mixture(experts=4, top_k=2, layers=(1,))},
        # Engram gates its lookup into each of mHC's residual streams.
        "engram": {"engram": Engram(layer_ids=(0,), num_embeddings=(88,), max_ngram_size=3, vocab_size=16,
                                    n_heads=2, head_dim=4, compressed_vocab_size=16),
                   "hyper_connections": HyperConnections(hc_mult=2, head="weighted")},
        "positions": {"position_embedding": "learned", "head_bias": True, "tie_embeddings": False},
        "own_gate": {"output_gate": True, "mixer": mixers.build("attention", gate_projection="own")},
        "fused_delta": {"layer_types": ("linear_attention", "full_attention"),
                        "kinds": {"linear_attention": {"mixer": delta}}},
        "shared_gate": {"mixture": Mixture(experts=4, top_k=2, layers=(1,), shared_features=8,
                                           shared_gate=True)},
    }
    def dory():
        # nGPT's step rates in a looped decoder that holds nGPT's every constraint: Dory's fixture's.
        from pathlib import Path

        from dew.interop import Pretrained

        fixture = Path(__file__).with_name("fixtures") / "hf" / "dory-tiny"
        return Pretrained.load(fixture, dtype="float32", attention_impl="reference").model, ids, {}

    return {**{f"decoder+{name}": lambda extra=extra: (tiny(max_seq_len=128, **extra), ids, {})
               for name, extra in fields.items()}, "decoder+ngpt": dory}


def _towers() -> dict:
    """The condition and media towers whose modules declare axes no catalogued model reaches: CLIP's text
    and vision towers and their projections, T5's encoder, Gemma 3n's MobileNet tower (its fixture's
    configuration), and DeepSeek V4's DSpark drafter (its fixture's)."""
    import json
    from pathlib import Path

    from dew.interop import Pretrained
    from dew.interop.hf_decoders import translate_wrapper_config
    from dew.nn.text_encoders import CLIP, CLIPTextTransformer, CLIPVisionTransformer, T5EncoderTransformer
    from dew.registry import towers

    fixtures = Path(__file__).with_name("fixtures") / "hf"
    ids = jnp.ones((1, 4), jnp.int32)
    text = functools.partial(CLIPTextTransformer, vocab_size=16, hidden_size=8, intermediate_size=16,
                             num_layers=2, num_heads=2, max_position_embeddings=8)
    vision = functools.partial(CLIPVisionTransformer, hidden_size=8, intermediate_size=16, num_layers=2,
                               num_heads=2, patch_size=2, image_size=4, num_channels=3)
    image = jnp.ones((1, 3, 4, 4))

    def mobilenet():
        config = json.loads((fixtures / "gemma3n-vision-tiny" / "config.json").read_text())
        record = translate_wrapper_config(config)
        return towers.from_record(record["tower"]).build(), (jnp.ones((1, 3, 64, 64)),), {}

    def dspark():
        model = Pretrained.load(fixtures / "deepseek-v4-dspark-tiny", weights=False, dtype="float32",
                                attention_impl="reference").model
        return model, (jnp.zeros((1, 8), jnp.int32),), {}

    return {
        "clip_text": lambda: (text(), (ids,), {}),
        "clip_vision": lambda: (vision(), (image,), {}),
        "clip": lambda: (CLIP(text(), vision(), projection_dim=4), (image, ids), {}),
        "t5_encoder": lambda: (T5EncoderTransformer(vocab_size=16, d_model=8, d_ff=16, num_layers=2,
                                                    num_heads=2, head_dim=6), (ids,), {}),
        "mobilenet_v5": mobilenet,
        "deepseek_v4_dspark": dspark,
    }


def cases() -> dict:
    """Each case's name, and how to build its module and the inputs it initializes on."""
    built = {}
    for architecture in [*sorted(models), *sorted(DECODERS)]:
        def model(architecture=architecture):
            inputs = tiny_inputs(architecture, jax.random.key(0))
            args, kwargs = inputs if isinstance(inputs[0], tuple) else (inputs, {})
            return build_model(architecture, "float32"), args, kwargs
        built[architecture] = model
    for kind in sorted(mixers.paths):
        def mixer(kind=kind):
            fields, overrides = KIND_FIELDS.get(kind, ({}, {}))
            return (tiny(mixer=mixers.build(kind, **fields), max_seq_len=128, **overrides),
                    (jnp.zeros((1, 8), jnp.int32),), {})
        built[f"mixer:{kind}"] = mixer
    return {**built, **_components(), **_features(), **_towers()}


TRAINED = ("causal_transformer", "causal_transformer+moe")
"""The cases whose trainer's state is recorded too: its Adam moments and EMA beside its parameters."""


def declared(variables, table=None) -> dict[str, tuple | None]:
    """Each leaf's declared logical axes, as the layout reads them, by its key path: those `variables`
    box, or the run's `table` (`Objective.parameter_axes`) for a state that mirrors them; None for a leaf
    no module declares, which the layout places by its shape."""
    rules = Layout().axis_rules
    # The EMA mirrors the variables whole under its field, as a trainer's state holds it.
    ema = (jax.tree_util.DictKey("ema"),)
    with boxed(ruled_boxes(boxed_axes(variables) if table is None else table, rules)):
        return {jax.tree_util.keystr(path): declared_axes(path[1:] if path[:1] == ema else path, leaf.ndim)
                for path, leaf in jax.tree_util.tree_leaves_with_path(unboxed(variables))}
