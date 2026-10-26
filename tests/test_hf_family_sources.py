"""Each registered family computes what its transformers source builds from the fields every family
shares, held by check 1 against the source run live over drawn inputs, from its float64 twin:

- YaRN (ledger 1594): every family whose rotary class builds its rope_type through
  `ROPE_INIT_FUNCTIONS` (transformers 5.16.1, modeling_rope_utils.py:668-674) reads a YaRN record,
  partial rotaries included, and its logits past the original context are the source's.
- The pad row (ledger 1593): a family whose token embedding is nn.Embedding(padding_idx=pad_token_id)
  gives the pad row no gradient from its lookups, and the rest of the table the source's gradient;
  a family whose source passes no padding_idx trains the row, as the source does.
"""
import contextlib
import json
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.traverse_util import flatten_dict
from interop_support import clearing_selections, token_draw, torch_logits
from reference_error import as_given, assert_as_exact

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
              "mimo-v2-flash-tiny": 8, "minimax-m2-tiny": 8}
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
                "mimo-v2-flash-tiny": 8, "llama4-tiny": 8, "granitemoe-tiny": 8}
"""Families whose source passes padding_idx, by fixture, with their declared check-1 seeds."""

UNPADDED_SEEDS = {"gpt-neox-tiny": 16, "falcon-tiny": 16}
"""Families whose source passes none (GPTNeoXModel's and FalconModel's plain nn.Embedding)."""


def _source(name: str, directory: Path, **fields) -> Path:
    """The fixture's config at `directory`, `fields` over its stored ones, and transformers' model of it
    with the probe's weights (`dew.interop.verify.scatter_weights`), whose norms sit near one: the
    stored tiny weights leave a hybrid decoder's gated norms at their epsilon, where both float32 arms
    round heavy-tailed."""
    from transformers import AutoConfig, AutoModelForCausalLM

    from dew.interop.verify import scatter_weights

    directory.mkdir(parents=True)
    config = {**json.loads((FIXTURES / name / "config.json").read_text()), **fields}
    (directory / "config.json").write_text(json.dumps(config))
    if getattr(AutoConfig.from_pretrained(directory), "head_dim", 0) is None:
        # The source reads a stated head_dim (MinistralAttention, `_compute_yarn_parameters`), which
        # MixtralConfig and MinistralConfig leave None where the width divides evenly.
        config["head_dim"] = config["hidden_size"] // config["num_attention_heads"]
        (directory / "config.json").write_text(json.dumps(config))
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

    config = json.loads((FIXTURES / name / "config.json").read_text())
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


def _embedding(params) -> tuple:
    """The path of the token table among a decoder's params."""
    (path,) = [path for path in flatten_dict(params) if path[-2:] == ("embed_tokens", "embedding")]
    return path


@pytest.mark.parametrize("name", sorted({**PADDED_SEEDS, **UNPADDED_SEEDS}))
def test_the_pad_row_takes_the_gradient_its_source_gives_it(name, tmp_path):
    """The family's token lookup is its source's: the loaded decoder pads the row the source's own
    token embedding module pads (its `padding_idx`, None where the source passes none), and over drawn
    ids, a quarter of them the pad id, under a drawn cotangent on the looked-up rows, the table's
    gradient holds check 1 against that module's lookup (`nn.Embedding.forward`, before any scale a
    subclass applies), the pad row's lookups adding nothing where it pads and training it where not.
    The lookup is where padding_idx acts; through a whole decoder its gradient is the rest of the
    network's too, which a gated delta net's float32 recurrence makes heavy-tailed in both arms."""
    import torch

    directory = _source(name, tmp_path / name, pad_token_id=PAD)
    loaded = Pretrained.load(directory, dtype="float32", attention_impl="reference")
    reference, wide = _reference(directory)
    padded = name in PADDED_SEEDS
    stated = PAD if padded else None
    assert loaded.model.padding_idx == reference.get_input_embeddings().padding_idx == stated
    path = _embedding(loaded.variables["params"])
    vocab, width = flatten_dict(loaded.variables["params"])[path].shape

    def draw(seed):
        rng = np.random.default_rng([0x1593, seed])
        ids = rng.integers(0, vocab, (8, 32))
        ids[rng.random(ids.shape) < 0.25] = PAD
        ids[0, 0] = PAD
        return ids.astype(np.int32), rng.standard_normal((*ids.shape, width)).astype(np.float32)

    @jax.jit
    def gradient(variables, ids, cotangent):
        def loss(params):
            looked_up = loaded.model.apply({**variables, "params": params}, ids, method="token_embeddings")
            return jnp.sum(looked_up * cotangent)
        return flatten_dict(jax.grad(loss)(variables["params"]))[path]

    @as_given
    def ours(ids, cotangent):
        return {"token table": np.asarray(gradient(loaded.variables, ids, cotangent), np.float64)}

    def theirs(source, precision=contextlib.nullcontext):
        def run(ids, cotangent):
            with precision():
                module = source.get_input_embeddings()
                module.zero_grad()
                rows = torch.nn.Embedding.forward(module, torch.from_numpy(ids).long())
                (rows * torch.from_numpy(cotangent).to(rows.dtype)).sum().backward()
                return {"token table": module.weight.grad.double().numpy()}
        return as_given(run)

    ids, cotangent = draw(0)
    pad_row = ours(ids, cotangent)["token table"][PAD]
    assert np.any(pad_row) != padded
    assert_as_exact(ours, theirs(reference), draw, f"{name} token lookup gradient",
                    seeds={**PADDED_SEEDS, **UNPADDED_SEEDS}[name], truth_fn=theirs(wide, float64))


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


def test_a_qwen35_wrapper_refuses_a_ramp_its_three_axis_rotary_does_not_apply(tmp_path):
    """Qwen 3.5's wrapper rotates image positions by three-axis tables of the plain frequencies, so a
    YaRN record on its text decoder is refused rather than dropped from them."""
    name = "qwen35-moe-native-tiny"
    config = json.loads((FIXTURES / name / "config.json").read_text())
    text = config["text_config"]
    text["rope_parameters"] = {**text["rope_parameters"], **YARN}
    shutil.copytree(FIXTURES / name, tmp_path / name)
    (tmp_path / name / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError):
        Pretrained.load(tmp_path / name, dtype="float32", attention_impl="reference")
