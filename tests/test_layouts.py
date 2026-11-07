"""Each family's one layout (`dew.interop.layouts`): its tensor names and its config read and written by the
same declaration, and a model its family's layout holds written from its own fields and read back as
itself (`source_layout`)."""

import json
from pathlib import Path

import numpy as np
import pytest
from model_support import flat_tree

from dew.interop import Pretrained, PretrainedDecoder, hf_decoders
from dew.interop.codecs import SIDECARS
from dew.interop.decoder_families import ENTRIES
from dew.interop.decoder_parts import ExportRefused
from dew.interop.layouts import (
    Choice,
    ConfigField,
    ConfigLayout,
    ConfigPart,
    Copy,
    Nested,
    Prefixed,
    Rebased,
    Restated,
    Route,
    Spelled,
    Table,
    When,
    source_layout,
)
from dew.interop.safetensors_io import header_specs

FIXTURES = Path(__file__).parent / "fixtures" / "hf"
ROUTES = (Route, Copy, Table, Spelled, Prefixed, Rebased, When)
CONFIGS = (ConfigLayout, ConfigPart, ConfigField, Nested, Restated, Choice)


def declared(node: object):
    """Every node of a family's route table or config layout, the node first: a nested record's, a
    restated layout's and both of a choice's among them."""
    yield node
    for child in (*getattr(node, "entries", ()), *getattr(node, "parts", ())):
        yield from declared(child)
    for attribute in ("routes", "route", "layout", "routed", "plain"):
        if hasattr(node, attribute):
            yield from declared(getattr(node, attribute))


def test_every_registered_family_reads_and_writes_by_its_one_layout():
    """Every registered family's tensor names are one route table (`Route` and its combinators) and its
    config one layout of parts each read and written together (`ConfigLayout`), from which both its reader
    and its writer derive: a family holds no reader or writer of its own beside them, the second inverse
    of a path, a config field or a weight a family once kept and drifted from (ledger 1026, 1411, 4562,
    4570, 4571)."""
    for family in ENTRIES:
        assert all(isinstance(node, ROUTES) for node in declared(family.routes)), family.model_types
        assert all(isinstance(node, CONFIGS) for node in declared(family.config)), family.model_types


def releases() -> list[str]:
    """The decoder releases among the fixtures, a registered family's config at their top and weights."""
    found = []
    for directory in sorted(FIXTURES.iterdir()):
        config = directory / "config.json"
        if not config.is_file() or not any(directory.glob("*.safetensors")):
            continue
        record = json.loads(config.read_text())
        if record.get("model_type") in hf_decoders.families() and "text_config" not in record:
            found.append(directory.name)
    return found


@pytest.mark.parametrize("name", releases())
def test_every_tensor_a_family_reads_it_writes_back_under_a_name_it_reads_into_the_same_leaf(name):
    """Each tensor of a release (each part of a packed one, `Packed`) its family reads into a leaf is written
    from that leaf under a name its routes read back into it: a release's own spelling, or the one its
    family writes where it reads two (DeepSeek V4's beside transformers', a bare model's beside its LM's)."""
    config = json.loads((FIXTURES / name / "config.json").read_text())
    names = [tensor for weights in sorted((FIXTURES / name).glob("*.safetensors"))
             for tensor in header_specs(weights)]
    assert names
    if config["model_type"] in ("diffusion_gemma", "diffusion_gemma_text"):
        # DiffusionGemma's denoiser holds its text decoder under its encoder and decoder, read by the text
        # family's names as its loader takes them (`hf_decoders._denoiser_sources`).
        from dew.interop.diffusion_gemma import text_config
        from dew.interop.sources import load_shards

        whole = config["model_type"] == "diffusion_gemma"
        config = text_config(config) if whole else config
        names = list(hf_decoders._denoiser_sources(names, load_shards(FIXTURES / name).__getitem__,
                                                   text_only=not whole)[0])
    family = hf_decoders.families()[config["model_type"]]
    record = hf_decoders.translate_config(config)
    for named in names:
        # A codec's sidecar (a weight's scales, its packed codes) is its weight's (`codecs.SIDECARS`).
        sidecar = next((suffix for suffix in SIDECARS if named.endswith(suffix)), None)
        stored = named if sidecar is None else named[:-len(sidecar)] + ".weight"
        packing = family.packing(stored)
        for part in ([stored] if packing is None
                     else [stored.removesuffix(packing.name) + piece for piece in packing.parts]):
            found = family.routes.read(part, record)
            assert found is not None, part
            if found.path is None:
                continue
            written = family.routes.write(found.path, record)
            assert written is not None, (part, found.path)
            again = family.routes.read(written, record)
            assert again is not None and again.path == found.path, (part, written)


WRITTEN = ("qwen3-moe-tiny", "qwen2-moe-tiny", "qwen3-next-tiny", "deepseek-v2-tiny", "deepseek-v3-tiny",
           "deepseek-v32-tiny", "glm4-moe-tiny", "glm-moe-dsa-tiny", "mixtral-tiny", "granitemoe-tiny",
           "mamba2-tiny", "nemotron-h-tiny", "nemotron-h-moe-tiny")
"""The routed and hybrid releases a model built in Dew writes from its own fields, which only a loaded
release's own bytes held before (export review, finding 1)."""


@pytest.mark.parametrize("name", WRITTEN)
def test_a_routed_or_hybrid_model_writes_its_release_from_its_own_fields_and_reads_back_as_itself(
        name, tmp_path):
    """A routed or hybrid decoder its family's layout holds (`source_layout`) is written from its fields
    alone (`PretrainedDecoder.from_model`): its config reads back as the same model, and every parameter
    and router state as the leaf it was written from."""
    loaded = Pretrained.load(str(FIXTURES / name), dtype="float32", attention_impl="reference")
    layout = source_layout(loaded.model)
    assert layout, layout.reason
    PretrainedDecoder.from_model(loaded.model, loaded.variables, tokenizer="byte").save(tmp_path)
    again = Pretrained.load(str(tmp_path), dtype="float32", attention_impl="reference")
    assert again.model == loaded.model
    for collection in ("params", "moe"):
        written = flat_tree(loaded.variables.get(collection, {}))
        read = flat_tree(again.variables.get(collection, {}))
        assert written.keys() == read.keys(), collection
        for path, leaf in written.items():
            assert np.array_equal(np.asarray(read[path]), np.asarray(leaf)), (collection, path)


def test_a_model_its_layout_cannot_hold_is_unwritable_with_its_reason():
    """A model no family's layout holds is `Unwritable`, falsy and naming why, before any file is written:
    a decoder with cross-attention, one whose family's config cannot state its computation (GPT-J's
    layout under a gated MLP, `hf_ungated`), and a model of no decoder class. A bundle of it is refused
    at once (`PretrainedDecoder.from_model` writes the layout `source_layout` answers), never built to be
    refused at its save."""
    from flax import linen as nn

    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.cross_attention import CrossAttention

    decoder = CausalTransformer(vocab_size=32, emb_features=16, num_layers=2, num_heads=2, num_kv_heads=2,
                                head_dim=8, mlp_features=32, max_seq_len=16)
    assert source_layout(decoder)
    gated_gptj = decoder.clone(parallel_residual=True, shared_parallel_norm=True, head_bias=True,
                               attention_bias=False, mlp="swiglu")
    for model in (decoder.clone(cross_attention=CrossAttention((1,), 4)), gated_gptj, nn.Dense(4)):
        layout = source_layout(model)
        assert not layout and layout.reason
        with pytest.raises(ExportRefused):
            PretrainedDecoder.from_model(model, {})
