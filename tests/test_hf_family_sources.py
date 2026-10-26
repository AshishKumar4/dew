"""Each registered family computes what its transformers source builds from the fields every family
shares, held by check 1 against the source run live over drawn inputs, from its float64 twin:

- YaRN (ledger 1594): every family whose rotary class builds its rope_type through
  `ROPE_INIT_FUNCTIONS` (transformers 5.16.1, modeling_rope_utils.py:668-674) reads a YaRN record,
  partial rotaries and Qwen 3.5's wrapper's three-axis rotary included, and its logits past the original
  context are the source's; Phi-3's config reads 'yarn' as LongRoPE, which its record is.
- The pad row (ledger 1593): a family whose token embedding is nn.Embedding(padding_idx=pad_token_id)
  gives the pad row no gradient from its lookups, and the rest of the table the source's gradient;
  a family whose source passes no padding_idx trains the row, as the source does. A remote source
  CI cannot run is the lookup its cited code builds.
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
from reference_error import assert_as_exact

from dew.interop import Pretrained
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
              "mimo-v2-flash-tiny": 8, "minimax-m2-tiny": 8, "qwen35-moe-text": 512}
"""Each fixture's declared check-1 seeds for its YaRN logits, sized by `seeds_for` on armada: Gemma 3n's
paired with the reference (`seeds_for(dew_fn=...)`), its arms sharing the draw's weights and inputs, where
the reference alone, heavy-tailed across draws, resolves within no 512."""

PAD = 3
"""The pad id each pad-row draw states and fills a quarter of its positions with."""

PADDED_SEEDS = {"llama-tiny": 8, "mistral-tiny": 8, "mixtral-tiny": 8, "ministral-tiny": 8,
                "qwen2-tiny": 8, "qwen2-moe-tiny": 8, "qwen3-tiny": 8, "qwen3-moe-tiny": 8,
                "qwen3-next-tiny": 8, "qwen35-tiny": 8, "glm4-moe-tiny": 8, "gemma-tiny": 8,
                "gemma2-tiny": 8, "gemma3-tiny": 8, "gemma3n-tiny": 8, "olmo3-tiny": 8,
                "starcoder2-tiny": 8, "phi-tiny": 8, "phi3-tiny": 8, "opt-tiny": 8, "gpt-oss-tiny": 8,
                "deepseek-v2-tiny": 8, "deepseek-v3-tiny": 8, "minimax-m2-tiny": 8,
                "mimo-v2-flash-tiny": 8, "llama4-tiny": 8, "granitemoe-tiny": 8,
                "deepseek-v32-tiny": 8, "deepseek-v4-tiny": 8, "diffusion-gemma-window-tiny": 8,
                "gemma4-moe-tiny": 8, "glm5-next-tiny": 8, "glm-moe-dsa-tiny": 8, "modernbert-tiny": 8,
                "muse-glimmer-text-tiny": 8, "qwen35-moe-text": 8, "kimi-k25-tiny": 8,
                "kimi-linear-tiny": 8, "kimi-k3-tiny": 8, "ouro-tiny": 8, "dream-tiny": 8,
                "llada-moe-tiny": 8, "llada-moe-v2-tiny": 8, "llada2-tiny": 8, "llada21-tiny": 8,
                "llada22-tiny": 8}
"""Families whose source passes padding_idx, by fixture, with their declared check-1 seeds."""

UNPADDED_SEEDS = {"gpt-neox-tiny": 16, "falcon-tiny": 16, "llada-tiny": 16, "mimo-v2-tiny": 16,
                  "k2-horizon-tiny": 16, "dory-tiny": 16, "deepseek-v41-tiny": 16}
"""Families whose source passes none (GPTNeoXModel's and FalconModel's plain nn.Embedding, and the
remote sources `REMOTE` cites)."""

REMOTE = {
    "kimi-k3-tiny": "modeling_kimi_linear.py:1093-1097 of moonshotai/Kimi-K3 at f831ab6",
    "ouro-tiny": "modeling_ouro.py:496-500 of ByteDance/Ouro-1.4B at 574fa66",
    "dream-tiny": "modeling_dream.py:602-605 of Dream-org/Dream-v0-Base-7B at 6572adb",
    "llada-moe-tiny": "modeling_lladamoe.py:895-898 of inclusionAI/LLaDA-MoE-7B-A1B-Base at daccaf1",
    "llada-moe-v2-tiny": "modeling_lladamoe_v2.py:1094-1097 of GSAI-ML/LLaDA-MoE-v2-30B-A3B-Base at 2a513f7",
    "llada2-tiny": "modeling_llada2_moe.py:750-754 of inclusionAI/LLaDA2.0-mini at dad945c",
    "llada21-tiny": "modeling_llada2_moe.py:761-765 of inclusionAI/LLaDA2.1-mini at 20e64e2",
    "llada22-tiny": "modeling_llada2_moe.py:770-774 of inclusionAI/LLaDA2.2-flash at 9b78da3",
    "llada-tiny": "modeling_llada.py:1035-1037 of GSAI-ML/LLaDA-8B-Base at 0f2787f, no padding_idx",
    "mimo-v2-tiny": "modeling_mimo_v2.py:1584 of XiaomiMiMo/MiMo-V2.6-Pro-MOPD at 6a7f202, no padding_idx",
    "k2-horizon-tiny": "modeling_k2_horizon.py:827-830 of IFM/K2-Horizon-0.9B at 383cb66, padding_idx from "
                       "a `padding_idx` field K2HorizonConfig does not declare",
    "dory-tiny": "vllm/model_executor/models/dory.py:232 at 55076950, a VocabParallelEmbedding",
    "deepseek-v41-tiny": "inference/model.py:152-156, :1201 of deepseek-ai/DeepSeek-V4.1-Flash at dba1be0, "
                         "a ParallelEmbedding",
}
"""Remote sources CI cannot run, by fixture, and where each builds its token embedding: an
nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id) where it pads, a lookup with no pad row
where it does not."""

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
    MixtralConfig and MinistralConfig leave None where the width divides evenly."""
    from transformers import AutoConfig

    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps(config))
    if getattr(AutoConfig.from_pretrained(directory), "head_dim", 0) is None:
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
    assert_as_exact(lambda ids: np.asarray(forward(loaded.variables, jnp.asarray(ids)), np.float64),
                    torch_logits(reference), draw, f"{name} YaRN logits", seeds=YARN_SEEDS[name],
                    truth_fn=torch_logits(wide, float64))


def _padded_config(name: str) -> dict:
    """The fixture's config with the pad id stated where its source's token embedding reads it."""
    config = _fixture_config(name)
    (config["text_config"] if name in TEXT_STATED else config)["pad_token_id"] = PAD
    return config


def _source_embedding(name: str, config: dict, directory: Path, table: np.ndarray):
    """The source's token embedding module over `config` (written at `directory`), holding `table`, and
    its float64 twin: the text model transformers builds from the config (`TEXT_MODELS`' class where no
    base model maps it), or for a remote source CI cannot run, the lookup its cited code builds
    (`REMOTE`)."""
    import importlib

    import torch
    from transformers import AutoConfig, AutoModel

    from tools.torch_precision import float64_twin

    if name in REMOTE:
        pad = PAD if name in PADDED_SEEDS else None

        def built(dtype):
            return torch.nn.Embedding(*table.shape, padding_idx=pad, dtype=dtype)
    else:
        settings = AutoConfig.from_pretrained(directory)
        text_model = TEXT_MODELS.get(config["model_type"])

        def built(dtype):
            if text_model is None:
                return AutoModel.from_config(settings, dtype=dtype)
            module, cls = text_model
            return getattr(importlib.import_module(module), cls)(settings).to(dtype)
    source = built(torch.float32)
    lookup = source if name in REMOTE else source.get_input_embeddings()
    with torch.no_grad():
        lookup.weight.copy_(torch.tensor(table))
    wide = float64_twin(source, lambda: built(torch.float64))
    return lookup, wide if name in REMOTE else wide.get_input_embeddings()


@pytest.mark.parametrize("name", sorted({**PADDED_SEEDS, **UNPADDED_SEEDS}))
def test_the_pad_row_takes_the_gradient_its_source_gives_it(name, tmp_path):
    """The family's token lookup is its source's (`assert_pads_as`): the decoder its reader builds from
    a config stating a pad id pads the row the source's own token embedding module pads, and trains it
    where the source does. The pad row is the reader's, decided where the config translates, and the
    lookup's gradient reads no weight, so the decoder is built from the config, which several of these
    sources state without a checkpoint or a causal LM to write one."""
    from dew.interop.hf_decoders import translate_config
    from dew.registry import models

    directory = tmp_path / name
    config = _padded_config(name) if name in REMOTE else _written(_padded_config(name), directory)
    model = models.build("causal_transformer",
                         {**translate_config(config), "dtype": "float32", "attention_impl": "reference"})
    variables = model.init(jax.random.key(0), jnp.zeros((1, 2), jnp.int32))
    lookup, wide = _source_embedding(name, config, directory, token_table(variables)[1])
    assert (lookup.padding_idx is not None) == (name in PADDED_SEEDS)
    placeholders = [config[field] for field in PLACEHOLDERS.get(name, ())]
    assert_pads_as(model, variables, lookup, wide, PAD, name,
                   seeds={**PADDED_SEEDS, **UNPADDED_SEEDS}[name],
                   looked_up=lambda ids: np.where(np.isin(ids, placeholders), 0, ids))


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

    assert_as_exact(lambda ids: np.asarray(forward(loaded.variables, jnp.asarray(ids)), np.float64),
                    torch_logits(reference), draw, f"{name} per-kind ramp logits",
                    seeds=KIND_RAMP_SEEDS[name], truth_fn=torch_logits(wide, float64))


WRAPPER = "qwen35-native-tiny"
"""A Qwen 3.5 wrapper fixture with its processor's outputs over prompts with images."""
WRAPPER_SEEDS = 512
"""The wrapper's YaRN logits' declared check-1 seeds, as many as qwen35-tiny's YaRN logits declare: the
gated delta net's float32 recurrence is heavy-tailed in both arms."""


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

    def ours(values):
        return np.asarray(forward(loaded.variables, processor.from_hf(values)), np.float64)[valid]

    def theirs(source, precision=contextlib.nullcontext):
        def run(values):
            with precision(), torch.no_grad():
                tensors = {name: torch.from_numpy(np.asarray(value)) for name, value in values.items()}
                logits = source(**tensors, use_cache=False).logits
            return logits.double().numpy()[valid]
        return run

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
    assert_as_exact(lambda ids: np.asarray(forward(loaded.variables, jnp.asarray(ids)), np.float64),
                    torch_logits(reference), draw, "phi3 'yarn' logits", seeds=8,
                    truth_fn=torch_logits(wide, float64))
