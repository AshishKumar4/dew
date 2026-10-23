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

############################################################################################################
# Text pooling reads the mask (T23)
############################################################################################################

SEEDS = {"pooled text conditioning": 16}
"""Check 1's declared seeds for each site of the file, sized by `seeds_for` (on armada)."""


def test_the_text_conditioning_projects_each_row_once_not_each_token():
    """The text reaches the adaLN vector only as the masked mean of its
    projected tokens. Projecting every token cost layout_parity's DiT (77
    text tokens of 1024 features for 16 image tokens) most of its step's
    matmul FLOPs, and a layout that splits no rows (tensor, sequence)
    repeated all of it on every device: 2.63 times one device's FLOPs on
    four 3090s. The vector is as exact as projecting every token and pooling
    them, a trained bias and an empty row included (check 1), and a longer
    text adds only its pooling's few operations per element, not a
    projection's. A draw is sixteen rows of 77 tokens, each real up to a
    length drawn from none to all, under weights and a bias drawn anew."""
    embed = ConditioningEmbed(emb_features=64, mlp_ratio=1)

    def draw(seed):
        keys = jax.random.split(jax.random.key(seed), 5)
        hidden = jax.random.normal(keys[0], (16, 77, 32))
        lengths = jax.random.randint(keys[1], (16,), 0, 78).at[0].set(0)
        mask = (jnp.arange(77)[None, :] < lengths[:, None]).astype(jnp.int32)
        temb = jax.random.uniform(keys[2], (16,), maxval=1000.0)
        params = embed.init(keys[3], temb, TextContext(hidden, mask))
        # A trained bias, which the pooled path adds once to each row's projection.
        params["params"]["text_context_proj"]["bias"] = jax.random.normal(keys[4], (64,))
        return params, temb, hidden, mask

    @jax.jit
    def pooled(params, temb, hidden, mask):
        return embed.apply(params, temb, TextContext(hidden, mask))

    @jax.jit
    def projected(params, temb, hidden, mask):
        projection = params["params"]["text_context_proj"]
        tokens = hidden @ projection["kernel"] + projection["bias"]
        return embed.apply(params, temb, None) + masked_mean(tokens, mask)

    label = "pooled text conditioning"
    assert_as_exact(pooled, projected, draw, label, seeds=SEEDS[label])


def test_padded_rows_do_not_move_the_conditioning_vector():
    """The tokenizer pads every prompt to 77 slots; what the padded rows hold
    must not reach the adaLN vector: the vector is the same, bit for bit,
    whatever they hold. Before the mask, they were averaged in and moved it.
    Once the mask says they are real, they move it."""
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
    counted = np.asarray(embed.apply(params, temb, TextContext(garbage, full)))
    assert np.isfinite(counted).all()
    assert not np.array_equal(counted, embed.apply(params, temb, TextContext(hidden, full)))


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
