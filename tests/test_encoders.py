"""The conditioning seam: an encoder's value reaches the model under one
keyword, and the text mask it carries is what the pooling weighs by, so a
padded row moves nothing.
"""


import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact

from dew.inputs import Condition, Field, InputSpec
from dew.nn.dit import ConditioningEmbed, TextContext, masked_mean


def test_text_pooling_excludes_nonfinite_padding_from_values_and_gradients():
    """Changing excluded text to NaN or infinity changes neither its pooled conditioning nor its
    gradient, including an entirely padded row."""
    values = jnp.arange(24, dtype=jnp.float32).reshape(2, 4, 3) / 8
    weights = jnp.asarray([[0.5, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])

    def loss(x):
        return jnp.sum(masked_mean(x, weights))

    clean = jax.value_and_grad(loss)(values)
    for filler in (jnp.nan, jnp.inf, -jnp.inf):
        poisoned = jnp.where(weights[:, :, None] != 0, values, filler)
        actual = jax.value_and_grad(loss)(poisoned)
        for observed, expected in zip(actual, clean, strict=True):
            np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize("axis", [None, 0, 1])
@pytest.mark.parametrize("weights", [[1.0, 0.0, 0.5], [[1.0, 0.0, 0.5]],
                                     [[1.0], [0.0], [1.0], [0.5]]])
def test_a_shared_mask_counts_every_row_in_a_weighted_mean(axis, weights):
    """A mask shared across the batch weighs as many rows as its explicit broadcast, including
    its denominator and the gradient of the pooled result."""
    from dew.rl.advantage import masked_mean as token_mean

    values = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
    shared = jnp.asarray(weights)
    explicit = jnp.broadcast_to(shared, values.shape)
    values = jnp.where(explicit != 0, values, jnp.nan)
    evaluate = jax.value_and_grad(lambda x, mask: jnp.sum(token_mean(x, mask, axis=axis)))
    for observed, expected in zip(evaluate(values, shared), evaluate(values, explicit), strict=True):
        np.testing.assert_array_equal(observed, expected)

############################################################################################################
# Text pooling reads the mask (T23)
############################################################################################################

PROJECTED_SEEDS = 16
"""The pooled-projection site's declared draws (`seeds_for`)."""


def test_the_text_conditioning_projects_each_row_once_not_each_token(rng):
    """The text reaches the adaLN vector only as the masked mean of its
    projected tokens. Projecting every token cost layout_parity's DiT (77
    text tokens of 1024 features for 16 image tokens) most of its step's
    matmul FLOPs, and a layout that splits no rows (tensor, sequence)
    repeated all of it on every device: 2.63 times one device's FLOPs on
    four 3090s. The vector is the same, a trained bias and an empty row
    included, and a longer text adds only its pooling's few operations per
    element, not a projection's: over drawn texts and times, the pooled
    projection is on par with projecting each token then pooling, from float64
    (check 1)."""
    embed = ConditioningEmbed(emb_features=64, mlp_ratio=1)
    mask = jnp.array([[1] * 20 + [0] * 57, [0] * 77, [1] * 77])
    params = embed.init(rng, jnp.ones(3), TextContext(jnp.ones((3, 77, 32)), mask))
    projection = params["params"]["text_context_proj"]
    projection["bias"] = projection["bias"] + 0.5

    def draw(seed):
        """A seed's text states and times, the trained parameters held."""
        texts, times = jax.random.split(jax.random.key(seed))
        return (params, jax.random.normal(texts, (3, 77, 32)),
                jax.random.uniform(times, (3,), minval=0.0, maxval=3.0))

    @jax.jit
    def pooled(params, hidden, temb):
        return embed.apply(params, temb, TextContext(hidden, mask))

    @jax.jit
    def each_token(params, hidden, temb):
        kernel = params["params"]["text_context_proj"]
        tokens = hidden @ kernel["kernel"] + kernel["bias"]
        return embed.apply(params, temb, None) + masked_mean(tokens, mask)

    assert_as_exact(pooled, each_token, draw, "pooled text projection", seeds=PROJECTED_SEEDS)


def test_padded_rows_do_not_move_the_conditioning_vector():
    """The tokenizer pads every prompt to 77 slots; what the padded rows hold
    must not reach the adaLN vector: they hold no term of its sum, so it is the
    same to the bit whatever they hold. Before the mask, they were averaged in
    and moved it."""
    embed = ConditioningEmbed(emb_features=16, mlp_ratio=1)
    key = jax.random.PRNGKey(0)
    hidden = jax.random.normal(jax.random.fold_in(key, 1), (2, 6, 8))
    mask = jnp.array([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]])
    temb = jnp.array([10.0, 20.0])
    params = embed.init(key, temb, TextContext(hidden, mask))

    baseline = embed.apply(params, temb, TextContext(hidden, mask))
    garbage = hidden.at[0, 3:].set(hidden[0, 3:] + 50.0)
    moved = embed.apply(params, temb, TextContext(garbage, mask))
    np.testing.assert_array_equal(moved, baseline)

    # the same rows do count once the mask says they are real
    full = jnp.ones_like(mask)
    assert not np.array_equal(embed.apply(params, temb, TextContext(garbage, full)),
                              embed.apply(params, temb, TextContext(hidden, full)))


def test_two_conditions_cannot_share_one_batch_field():
    """The objective reads batch[condition.field] under each keyword, so two
    conditions on one field would tokenize twice into one key and the second
    writing would win. A run with two text towers names a field per tower."""
    from dew.inputs import CharTable

    conditions = {
        "clip": Condition(CharTable.from_pretrained(seed=0), field="text"),
        "t5": Condition(CharTable.from_pretrained(seed=1), field="text"),
    }
    with pytest.raises(ValueError):
        InputSpec(sample=Field("image", (8, 8, 3)), conditions=conditions)


def test_condition_roundtrip_preserves_nondefault_character_embeddings():
    from dew.inputs import CharTable

    condition = Condition(CharTable.from_pretrained(
        seed=7, dtype="bfloat16", tokens=4, features=2, vocab=8))
    restored = Condition.from_json(condition.to_json())
    contexts = [value.encoder.encode(value.encoder.params, value.encoder.tokenize(["ab", ""]))
                for value in (condition, restored)]
    np.testing.assert_array_equal(contexts[0].hidden, contexts[1].hidden)
    np.testing.assert_array_equal(contexts[0].mask, contexts[1].mask)
    assert contexts[1].hidden.dtype == jnp.bfloat16


def swap_two_prompt_tokens(folder, prompt: str) -> None:
    """Give the first ordinary token `prompt` tokenizes to and the next ordinary one each other's id,
    in every vocabulary file of the tokenizer in `folder`: the same text then tokenizes to other
    ids."""
    import json

    from dew.data.text import load_tokenizer

    tokenizer = load_tokenizer(str(folder))
    special = set(tokenizer.all_special_ids)
    first = next(index for index in tokenizer(prompt)["input_ids"] if index not in special)
    second = next(index for index in range(first + 1, len(tokenizer)) if index not in special)
    for name in ("tokenizer.json", "vocab.json"):
        path = folder / name
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        vocab = data["model"]["vocab"] if name == "tokenizer.json" else data
        if isinstance(vocab, list):
            vocab[first], vocab[second] = vocab[second], vocab[first]
        else:
            named = {index: token for token, index in vocab.items()}
            vocab[named[first]], vocab[named[second]] = second, first
        path.write_text(json.dumps(data))


@pytest.mark.parametrize("family, archive, inside, tokenizer", [
    ("sd", "tiny_diffusers", "sd", "tokenizer"),
    ("sd3", "sd3_source", "pipeline", "tokenizer_3"),
    ("z_image", "z_image_source", "pipeline", "tokenizer"),
    ("wan", "wan_pipeline", "pipeline", "tokenizer"),
])
def test_a_pipelines_conditioner_records_the_tokenizer_files_it_reads(tmp_path, family, archive, inside,
                                                                       tokenizer):
    """A published pipeline's text conditioner tokenizes with the tokenizer files it finds and
    records their hashes, each of its towers' (an SD3's T5 as well as its CLIPs): a pipeline loaded
    again in the same process after a tokenizer's vocabulary moved two ids tokenizes a caption to
    other ids, and the identity an exact resume checks those ids against moves with them."""
    import tarfile
    from pathlib import Path

    from dew.interop.pretrained import load_diffusion_source

    with tarfile.open(Path(__file__).parent / "fixtures" / f"{archive}.tar.xz") as opened:
        opened.extractall(tmp_path, filter="data")
    directory = tmp_path / inside

    def loaded():
        source = load_diffusion_source(str(directory), dtype="float32", attention_impl="xla")
        encoder = source.inputs.conditions["conditioning"].encoder
        return [np.asarray(ids) for ids in jax.tree.leaves(encoder.tokenize(["a red bird"]))], \
            encoder.dependencies()

    (ids, recorded) = loaded()
    swap_two_prompt_tokens(directory / tokenizer, "a red bird")
    (moved_ids, moved) = loaded()
    assert any(not np.array_equal(old, new) for old, new in zip(ids, moved_ids, strict=True)), family
    assert moved != recorded, family
