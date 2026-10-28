"""Each registered family computes what its transformers source builds from the fields every family
shares, held by check 1 against the source run live over drawn inputs, from its float64 twin:

- YaRN (ledger 1594): every family whose rotary class builds its rope_type through
  `ROPE_INIT_FUNCTIONS` (transformers 5.16.1, modeling_rope_utils.py:668-674) reads a YaRN record,
  partial rotaries and Qwen 3.5's wrapper's three-axis rotary included, and its logits past the original
  context are the source's; Phi-3's config reads 'yarn' as LongRoPE, which its record is.
- The pad row (ledger 1593): a family whose token embedding is nn.Embedding(padding_idx=pad_token_id)
  gives the pad row no gradient from its lookups, and the rest of the table the source's gradient;
  a family whose source passes no padding_idx trains the row, as the source does, with the pad id
  stated, left out and null. A source whose own code CI does not run is held to a recording of it
  (`RECORDED`).
"""
import contextlib
import json
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from interop_support import assert_pads_as, clearing_selections, token_draw, token_table, torch_logits
from reference_error import as_given, assert_as_exact

from dew.interop import Pretrained
from tools.reference_source import PAD
from tools.torch_precision import float64

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf"

YARN = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 16, "beta_fast": 16.0,
        "beta_slow": 2.0}
"""A YaRN record whose ramp bends inside a tiny head's frequencies: 16 original positions stretched four
times, its correction range moved off the defaults so both ends of the blend are exercised."""
CONTEXT = 64
ROWS, LENGTH = 8, 48
"""Each draw's rows and tokens: past the original 16 positions, within the 64 the factor reaches."""

YARN_SEEDS = {"qwen2-tiny": 8, "qwen2-moe-tiny": 8, "qwen3-moe-tiny": 8, "qwen3-next-tiny": 128,
              "qwen35-tiny": 512, "glm4-moe-tiny": 8, "gpt-neox-tiny": 8, "falcon-tiny": 8,
              "starcoder2-tiny": 8, "mistral-tiny": 8, "mixtral-tiny": 8, "ministral-tiny": 8,
              "gemma-tiny": 8, "gemma2-tiny": 8, "gemma3-tiny": 8, "gemma3n-tiny": 512,
              "mimo-v2-flash-tiny": 8, "minimax-m2-tiny": 8, "qwen35-moe-text": 16}
"""Each fixture's declared check-1 seeds for its YaRN logits, sized by `seeds_for` on armada: Gemma 3n's
and Qwen 3.5 MoE's text decoder's paired with the reference (`seeds_for(dew_fn=...)`), their arms sharing
the draw's weights and inputs, where the reference alone, heavy-tailed across draws, resolves within no
512."""

PADDED = {"llama-tiny", "mistral-tiny", "mixtral-tiny", "ministral-tiny",
                "qwen2-tiny", "qwen2-moe-tiny", "qwen3-tiny", "qwen3-moe-tiny",
                "qwen3-next-tiny", "qwen35-tiny", "glm4-moe-tiny", "gemma-tiny",
                "gemma2-tiny", "gemma3-tiny", "gemma3n-tiny", "olmo3-tiny",
                "starcoder2-tiny", "phi-tiny", "phi3-tiny", "opt-tiny", "gpt-oss-tiny",
                "deepseek-v2-tiny", "deepseek-v3-tiny", "minimax-m2-tiny",
                "mimo-v2-flash-tiny", "llama4-tiny", "granitemoe-tiny",
                "deepseek-v32-tiny", "deepseek-v4-tiny", "diffusion-gemma-window-tiny",
                "gemma4-moe-tiny", "glm5-next-tiny", "glm-moe-dsa-tiny", "modernbert-tiny",
                "muse-glimmer-text-tiny", "qwen35-moe-text", "kimi-k25-tiny",
                "kimi-linear-tiny", "kimi-k3-tiny", "ouro-tiny", "dream-tiny",
                "llada-moe-tiny", "llada-moe-v2-tiny", "llada2-tiny", "llada21-tiny",
                "llada22-tiny", "lfm3-tiny"}
"""Families whose source pads a stated pad id, by fixture."""

UNPADDED = {"gpt-neox-tiny", "falcon-tiny", "llada-tiny", "mimo-v2-tiny", "k2-horizon-tiny",
            "deepseek-v41-tiny"}
"""Families whose source pads none (GPTNeoXModel's and FalconModel's plain nn.Embedding, and the
recorded sources `RECORDED` names). Dory's only source is vLLM's inference code, which runs no
backward and whose VocabParallelEmbedding takes no pad id (vllm/model_executor/models/dory.py:232 at
55076950), so it gives the pad row no gradient to match, and Dew trains it as any row."""

PAD_SEEDS = {True: 8, False: 16}
"""The declared check-1 seeds of a lookup's gradient, by whether the lookup pads a row."""

RECORDED = {
    "kimi-linear-tiny": "tools/kimi_linear_reference.py padding",
    "kimi-k3-tiny": "tools/kimi_k3_reference.py padding",
    "ouro-tiny": "tools/ouro_reference.py --padding",
    "k2-horizon-tiny": "tools/k2_horizon_reference.py --padding",
    "deepseek-v41-tiny": "tools/deepseek_v41_reference.py --padding",
    **dict.fromkeys(("llada-tiny", "dream-tiny", "mimo-v2-tiny", "lfm3-tiny"),
                    "tools/remote_code_reference.py --padding"),
    **dict.fromkeys(("llada-moe-tiny", "llada-moe-v2-tiny", "llada2-tiny", "llada21-tiny", "llada22-tiny"),
                    "tools/llada_moe_reference.py --padding"),
}
"""Sources whose own code (its release's modeling files at their pinned commits, under the transformers
each runs on) CI does not run, by fixture, with the writer that ran it and recorded, in PADDING beside
the fixture, the rows its token lookup leaves untrained in each of PADDING_CASES
(`reference_source.record_padding`)."""

TEXT_STATED = {"kimi-k25-tiny", "kimi-k3-tiny"}
"""Wrappers whose language model is built from their text_config, whose pad id it reads
(modeling_kimi_k25.py:586; modeling_kimi_k3.py:919 of moonshotai/Kimi-K3 at f831ab6): the draw states
it there."""

PLACEHOLDERS = {"kimi-k25-tiny": ("image_token_id", "video_token_id")}
"""A wrapper whose model looks its media placeholder ids up as token zero before its table
(modeling_kimi_k25.py:682-685), the ids its config's fields name, as Dew's lookup does
(`CausalTransformer.embedding_zero_ids`)."""

TEXT_OF = {"qwen35-moe-text": "qwen35-moe-native-tiny"}
"""A text decoder no fixture holds alone, read from a wrapper fixture's text_config."""

TEXT_MODELS = {
    "diffusion_gemma_text": ("transformers.models.diffusion_gemma", "DiffusionGemmaEncoderTextModel")}
"""A text config transformers maps to no base model, and the model its source builds from it (its
decoder's embedding is the same, from the same field, modeling_diffusion_gemma.py:1212-1219)."""


def _fixture_config(name: str) -> dict:
    """The fixture's stored config, a wrapper fixture's text_config for a text decoder `TEXT_OF` names."""
    config = json.loads((FIXTURES / TEXT_OF.get(name, name) / "config.json").read_text())
    return config["text_config"] if name in TEXT_OF else config


def _written(config: dict, directory: Path) -> dict:
    """`config` written at `directory` as its source builds from it, a head_dim its class leaves None
    stated: the source reads a stated one (MinistralAttention, `_compute_yarn_parameters`), which
    MixtralConfig and MinistralConfig leave None where the width divides evenly. The class's fields are
    read as it serializes them, since Gemma 4's refuses a read of its per-layer head_dim."""
    from transformers import AutoConfig

    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps(config))
    if AutoConfig.from_pretrained(directory).to_dict().get("head_dim", 0) is None:
        config = {**config, "head_dim": config["hidden_size"] // config["num_attention_heads"]}
        (directory / "config.json").write_text(json.dumps(config))
    return config


def _source(name: str, directory: Path, **fields) -> Path:
    """The fixture's config at `directory`, `fields` over its stored ones (`_written`), and transformers'
    model of it with the probe's weights (`dew.interop.verify.scatter_weights`), whose norms sit near
    one: the stored tiny weights leave a hybrid decoder's gated norms at their epsilon, where both
    float32 arms round heavy-tailed."""
    from transformers import AutoConfig, AutoModelForCausalLM

    from dew.interop.verify import scatter_weights

    _written({**_fixture_config(name), **fields}, directory)
    model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(directory))
    scatter_weights(model)
    model.save_pretrained(directory)
    return directory


def _yarned(config: dict) -> dict:
    """`config`'s rotary as a YaRN record in its own spelling: the full layers' record of a config that
    keeps one per layer type, else its one record, its base and partial fraction kept."""
    parameters = config.get("rope_parameters") or {"rope_theta": config.get("rope_theta", 10000.0)}
    if set(parameters) <= {"full_attention", "sliding_attention"}:
        rope = {**parameters, "full_attention": {**parameters["full_attention"], **YARN}}
    else:
        rope = {**parameters, **YARN}
    return {"rope_parameters": rope, "max_position_embeddings": CONTEXT}


def _reference(directory: Path):
    """The source's causal LM over `directory` in float32 with eager attention and experts (the
    grouped product multiplies no float64), and its float64 twin."""
    import torch
    from transformers import AutoModelForCausalLM

    from tools.torch_precision import float64_twin

    def built(dtype):
        return AutoModelForCausalLM.from_pretrained(directory, dtype=dtype, attn_implementation="eager",
                                                    experts_implementation="eager")
    model = built(torch.float32)
    return model.eval(), float64_twin(model, lambda: built(torch.float64)).eval()


@pytest.mark.parametrize("name", sorted(YARN_SEEDS))
def test_a_yarn_record_extends_every_family_whose_source_builds_it(name, tmp_path):
    """The family's reader takes a YaRN record onto its rotated layers, and its logits over drawn ids
    past the original context hold check 1 against the source's."""

    config = _fixture_config(name)
    directory = _source(name, tmp_path / name, **_yarned(config))
    loaded = Pretrained.load(directory, dtype="float32", attention_impl="reference")
    model = loaded.model
    assert model.yarn is not None or any(kind.yarn is not None for kind in (model.kinds or {}).values())
    reference, wide = _reference(directory)
    forward = jax.jit(model.apply)

    draw = clearing_selections(loaded, token_draw(0x1594, config["vocab_size"], (ROWS, LENGTH)), name)

    # The twin runs inside `float64`, which widens the casts to float32 its code makes (a norm's,
    # a recurrence's), so its logits are float64's.
    assert_as_exact(as_given(lambda ids: np.asarray(forward(loaded.variables, jnp.asarray(ids)), np.float64)),
                    torch_logits(reference), draw, f"{name} YaRN logits", seeds=YARN_SEEDS[name],
                    truth_fn=torch_logits(wide, float64))


def _padding_case(name: str, case: str, directory: Path) -> tuple[dict, dict | None]:
    """The fixture's config with its pad id as `case` puts it, where its source's token embedding reads it
    (`padding_variant`), its vocabulary grown to hold a pad id the source resolves past the table (its
    config class's default, where the config states none), and, where `RECORDED` names a recording of
    the source, the case it recorded: the source's table's rows and those its lookup leaves untrained.
    Another source's config is written at `directory` (`_written`), which transformers builds the source
    from, and its case is None."""
    from transformers import AutoConfig

    from tools.reference_source import PADDING, padding_variant

    place = ("text_config",) if name in TEXT_STATED else ()
    if name in RECORDED:
        recorded = json.loads((FIXTURES / name / PADDING).read_text())
        assert recorded["place"] == list(place)
        stated = recorded["cases"][case]
        return padding_variant(_fixture_config(name), case, place, stated["vocab_size"]), stated
    config = _written(padding_variant(_fixture_config(name), case, place), directory)
    resolved = AutoConfig.from_pretrained(directory)
    level = config
    for key in place:
        resolved, level = getattr(resolved, key), level[key]
    pad = getattr(resolved, "pad_token_id", None)
    if pad is not None and pad >= level["vocab_size"]:
        shutil.rmtree(directory)
        config = _written(padding_variant(_fixture_config(name), case, place, pad + 1), directory)
    return config, None


def _source_embedding(name: str, config: dict, directory: Path, table: np.ndarray, untrained):
    """The source's token embedding module over `config` (written at `directory`), holding `table`, and
    its float64 twin: the text model transformers builds from the config (`TEXT_MODELS`' class where no
    base model maps it), or, where a recording of the source's own code gives the rows its lookup leaves
    `untrained`, the nn.Embedding padding the one it recorded."""
    import importlib

    import torch
    from transformers import AutoConfig, AutoModel

    from tools.torch_precision import float64_twin

    if untrained is not None:
        assert len(untrained) <= 1

        def built(dtype):
            return torch.nn.Embedding(*table.shape, padding_idx=next(iter(untrained), None), dtype=dtype)
    else:
        settings = AutoConfig.from_pretrained(directory)
        text_model = TEXT_MODELS.get(config["model_type"])

        def built(dtype):
            if text_model is None:
                return AutoModel.from_config(settings, dtype=dtype)
            module, cls = text_model
            return getattr(importlib.import_module(module), cls)(settings).to(dtype)
    source = built(torch.float32)
    lookup = source if untrained is not None else source.get_input_embeddings()
    with torch.no_grad():
        lookup.weight.copy_(torch.tensor(table))
    wide = float64_twin(source, lambda: built(torch.float64))
    return lookup, wide if untrained is not None else wide.get_input_embeddings()


def _untrained(gradient) -> list[int]:
    """The rows of a table's gradient that are zero throughout."""
    return np.flatnonzero(~np.asarray(gradient).any(axis=-1)).tolist()


@pytest.mark.parametrize("case", ("stated", "omitted", "null"))
@pytest.mark.parametrize("name", sorted(PADDED | UNPADDED))
def test_the_pad_row_takes_the_gradient_its_source_gives_it(name, case, tmp_path):
    """The family's token lookup is its source's (`assert_pads_as`), with the config's pad id stated, left
    out (the source's config class's default) and null: the decoder its reader builds leaves untrained
    the rows the source's own token embedding leaves untrained when each row is looked up once, and its
    gradient under drawn ids and cotangents holds check 1 against the source's. The pad row is the
    reader's, decided where the config translates, and the lookup's gradient reads no weight, so the
    decoder is built from the config, which several of these sources state without a checkpoint or a
    causal LM to write one."""
    import torch

    from dew.interop.hf_decoders import translate_config
    from dew.registry import models

    directory = tmp_path / name
    config, recorded = _padding_case(name, case, directory)
    model = models.build("causal_transformer",
                         {**translate_config(config), "dtype": "float32", "attention_impl": "reference"})
    variables = model.init(jax.random.key(0), jnp.zeros((1, 2), jnp.int32))
    table = token_table(variables)[1]
    rows = jnp.arange(table.shape[0])[None]
    placeholders = [config[field] for field in PLACEHOLDERS.get(name, ())]

    def looked_up(ids):
        return np.where(np.isin(ids, placeholders), 0, ids)

    if recorded is None:
        lookup, wide = _source_embedding(name, config, directory, table, None)
        torch.nn.Embedding.forward(lookup, torch.from_numpy(looked_up(np.asarray(rows[0])))).sum().backward()
        untrained = _untrained(lookup.weight.grad.numpy())
    else:
        assert table.shape[0] == recorded["rows"]
        untrained = recorded["untrained"]
        lookup, wide = _source_embedding(name, config, directory, table, untrained)
    looked = jax.grad(lambda params: jnp.sum(
        model.apply({**variables, "params": params}, rows, method="token_embeddings")))(variables["params"])
    assert _untrained(token_table({"params": looked})[1]) == untrained
    if case == "stated":
        assert (lookup.padding_idx is not None) == (name in PADDED)
    padded = lookup.padding_idx is not None
    assert_pads_as(model, variables, lookup, wide, lookup.padding_idx if padded else PAD, f"{name} {case}",
                   seeds=PAD_SEEDS[padded], looked_up=looked_up)


KIND_RAMPS = {
    # Full layers under YaRN with fractional correction bounds and a record that says not to
    # truncate them, sliding layers under Llama 3.1's ramp: each kind takes its own, and a nested
    # record's YaRN truncates (modeling_rope_utils.py:474 reads the top-level dict).
    "gemma3-tiny": {"full_attention": {**YARN, "beta_fast": 13.0, "beta_slow": 1.5, "truncate": False},
                    "sliding_attention": {"rope_type": "llama3", "factor": 4.0, "low_freq_factor": 1.0,
                                          "high_freq_factor": 2.0, "original_max_position_embeddings": 16}},
    # A YaRN record without its fraction turns the whole head, as `_compute_yarn_parameters` does,
    # where MiMo's plain record defaults to a third of it.
    "mimo-v2-flash-tiny": {"full_attention": YARN},
}
KIND_RAMP_SEEDS = {"gemma3-tiny": 8, "mimo-v2-flash-tiny": 8}
"""Each fixture's declared check-1 seeds for its per-kind ramps' logits, sized by `seeds_for` on armada."""


@pytest.mark.parametrize("name", sorted(KIND_RAMPS))
def test_each_layer_kind_rotates_by_its_own_record(name, tmp_path):
    """A config whose layer kinds state their own rope records loads each kind's ramp as the source's
    initializer for that kind computes it, and its logits past the original context hold check 1
    against the source's."""

    config = json.loads((FIXTURES / name / "config.json").read_text())
    rope = {kind: {"rope_theta": record["rope_theta"], **KIND_RAMPS[name].get(kind, {})}
            for kind, record in config["rope_parameters"].items()}
    directory = _source(name, tmp_path / name, rope_parameters=rope, max_position_embeddings=CONTEXT)
    loaded = Pretrained.load(directory, dtype="float32", attention_impl="reference")
    reference, wide = _reference(directory)
    forward = jax.jit(loaded.model.apply)

    draw = clearing_selections(loaded, token_draw(0x15940, config["vocab_size"], (ROWS, LENGTH)), name)

    assert_as_exact(as_given(lambda ids: np.asarray(forward(loaded.variables, jnp.asarray(ids)), np.float64)),
                    torch_logits(reference), draw, f"{name} per-kind ramp logits",
                    seeds=KIND_RAMP_SEEDS[name], truth_fn=torch_logits(wide, float64))


WRAPPER = "qwen35-native-tiny"
"""A Qwen 3.5 wrapper fixture with its processor's outputs over prompts with images."""
WRAPPER_SEEDS = 8
"""The wrapper's YaRN logits' declared check-1 seeds, sized by `seeds_for` on armada paired with the
reference (`seeds_for(dew_fn=...)`): the reference alone, heavy-tailed across draws through the gated
delta net's float32 recurrence, resolves within no 512."""


def test_a_qwen35_wrapper_turns_its_three_axis_rotary_by_a_yarn_record(tmp_path):
    """Qwen 3.5's wrapper turns each rotated pair of an image token by its temporal, height or width id
    (`mrope_section`) at the frequencies its record builds through `ROPE_INIT_FUNCTIONS`, and scales
    cos and sin by the record's attention factor (Qwen3_5TextRotaryEmbedding, modeling_qwen3_5.py:143-213):
    under a YaRN record, its logits over the fixture's prompts with images, their text ids and pixels
    drawn, hold check 1 against the source's."""
    import torch
    from transformers import AutoModelForImageTextToText

    from tools.torch_precision import float64_twin

    directory = shutil.copytree(FIXTURES / WRAPPER, tmp_path / WRAPPER)
    config = json.loads((directory / "config.json").read_text())
    text = config["text_config"]
    text.update(rope_parameters={**text["rope_parameters"], **YARN}, max_position_embeddings=CONTEXT)
    (directory / "config.json").write_text(json.dumps(config))
    loaded = Pretrained.load(directory, dtype="float32", attention_impl="reference")
    assert loaded.processor is not None

    def built(dtype):
        return AutoModelForImageTextToText.from_pretrained(directory, dtype=dtype,
                                                           attn_implementation="eager")
    reference = built(torch.float32).eval()
    wide = float64_twin(reference, lambda: built(torch.float64)).eval()

    held = {name: np.load(directory / f"{name}.npy")
            for name in ("input_ids", "attention_mask", "image_grid_thw", "mm_token_type_ids")}
    pixels = np.load(directory / "pixel_values.npy").shape
    valid = held["attention_mask"].astype(bool)
    # The text a prompt holds, below the image and vision marker ids.
    words = valid & (held["input_ids"] < config["image_token_id"])

    def draw(seed):
        rng = np.random.default_rng([0x15942, seed])
        ids = np.where(words, rng.integers(0, config["image_token_id"], words.shape), held["input_ids"])
        return ({**held, "input_ids": ids, "pixel_values": rng.standard_normal(pixels).astype(np.float32)},)

    processor = loaded.processor
    model = loaded.model
    forward = jax.jit(lambda variables, inputs: model.apply(variables, inputs.tokens, **inputs.kwargs()))

    @as_given
    def ours(values):
        return np.asarray(forward(loaded.variables, processor.from_hf(values)), np.float64)[valid]

    def theirs(source, precision=contextlib.nullcontext):
        def run(values):
            with precision(), torch.no_grad():
                tensors = {name: torch.from_numpy(np.asarray(value)) for name, value in values.items()}
                logits = source(**tensors, use_cache=False).logits
            return logits.double().numpy()[valid]
        return as_given(run)

    assert_as_exact(ours, theirs(reference), draw, "Qwen 3.5 wrapper YaRN logits", seeds=WRAPPER_SEEDS,
                    truth_fn=theirs(wide, float64))


def test_a_phi3_record_spelled_yarn_is_the_longrope_its_source_reads(tmp_path):
    """Phi3Config reads rope_type 'yarn' as LongRoPE's older spelling (configuration_phi3.py:101-104) and
    refuses any other ramp (:122-123), so Phi-3's rotary builds no YaRN: a Phi-3 record spelled 'yarn'
    is the source's LongRoPE, and its logits past the original context hold check 1 against it."""
    config = _fixture_config("phi3-tiny")
    rope = {**config["rope_parameters"], "rope_type": "yarn"}
    directory = _source("phi3-tiny", tmp_path / "phi3", rope_parameters=rope)
    # The source writes its config back as it reads it, LongRoPE by name; the release's spelling is put back.
    saved = json.loads((directory / "config.json").read_text())
    (directory / "config.json").write_text(json.dumps({**saved, "rope_parameters": rope}))
    loaded = Pretrained.load(directory, dtype="float32", attention_impl="reference")
    reference, wide = _reference(directory)
    forward = jax.jit(loaded.model.apply)
    draw = token_draw(0x15943, config["vocab_size"], (ROWS, config["max_position_embeddings"]))
    assert_as_exact(as_given(lambda ids: np.asarray(forward(loaded.variables, jnp.asarray(ids)), np.float64)),
                    torch_logits(reference), draw, "phi3 'yarn' logits", seeds=8,
                    truth_fn=torch_logits(wide, float64))
