"""The recipe chain: SFT then DPO sharing one decoder, linked checkpoints.

A `Recipe` runs its stages in order, each in its own directory, with every
stage after the first initializing from the previous stage's final
parameters. The observable link: the DPO stage's frozen reference, written
at its first step, is the SFT stage's final parameters byte for byte. A
stage that names the wrong data, an unknown loss, or a GRPO stage without a
reward fails before anything trains.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import linen as nn

from dew.data import ChatMessages, Loading, PreferencePairs, Prompts
from dew.training import Layout

from recipes.chain import Recipe, Stage

TOKENIZER = Path(__file__).resolve().parent / "fixtures" / "tokenizers" / "tiny-chat"


class TinyHead(nn.Module):
    """A position-wise map with the backbone's scoring contract: int32 ids
    in, float32 logits out, the head split off behind `hidden_states` and
    `head_weight`."""

    vocab_size: int
    final_logit_softcap = None
    precision = None

    def setup(self):
        self.lm_head = nn.Dense(self.vocab_size, use_bias=False)

    @nn.compact
    def hidden_states(self, tokens, train: bool = False, **packing):
        x = nn.Embed(self.vocab_size, 8)(tokens)
        h = nn.LayerNorm()(x)
        return nn.LayerNorm()(x + nn.Dense(8)(nn.gelu(nn.Dense(16)(h))))

    def __call__(self, tokens, train: bool = False):
        return self.lm_head(
            self.hidden_states(tokens, train=train)).astype(jnp.float32)

    def head_weight(self, params):
        return params["lm_head"]["kernel"].astype(jnp.float32)


def chat_parquet(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    conversation = [{"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"}]
    path = tmp_path / "chat.parquet"
    pq.write_table(pa.table({"prompt": [conversation] * 16}), path)
    return str(path)


def pair_records():
    pairs = [
        {"chosen": [1, 2, 3, 4], "rejected": [1, 2, 5],
         "chosen_mask": [0, 0, 1, 1], "rejected_mask": [0, 0, 1]},
        {"chosen": [2, 3, 4, 1], "rejected": [2, 3, 6],
         "chosen_mask": [0, 0, 1, 1], "rejected_mask": [0, 0, 1]},
    ]
    return tuple(json.dumps(pair) for pair in pairs * 4)


def stages(tmp_path):
    return (
        Stage(name="sft", data=ChatMessages(
            tokenizer=str(TOKENIZER), path=chat_parquet(tmp_path), seq_len=31,
            loading=Loading(workers=0)), steps=1),
        Stage(name="dpo", data=PreferencePairs(
            records=pair_records(), seq_len=4, loading=Loading(workers=0)),
            steps=1),
    )


def test_each_stage_continues_the_last(tmp_path):
    """Two one-step stages: both checkpoint directories land, the second
    stage trains (its parameters move off the first stage's), and its frozen
    reference is the first stage's final tree."""
    recipe = Recipe(TinyHead(vocab_size=8), optax.sgd(0.1), jax.random.key(0),
                    stages(tmp_path), str(tmp_path / "chain"), batch=8,
                    layout=Layout(min_shard=1, tolerance=1.0))

    states = recipe.run()

    assert len(states) == 2
    assert (tmp_path / "chain" / "sft").is_dir()
    assert (tmp_path / "chain" / "dpo").is_dir()
    first = [leaf.tobytes() for leaf in jax.tree.leaves(states[0].params)]
    frozen = [leaf.tobytes() for leaf in jax.tree.leaves(states[1].ema)]
    assert frozen == first
    second = [leaf.tobytes() for leaf in jax.tree.leaves(states[1].params)]
    assert any(a != b for a, b in zip(first, second, strict=True))


def test_a_mislinked_chain_is_refused(tmp_path):
    model, optim, key = TinyHead(vocab_size=8), optax.sgd(0.1), jax.random.key(0)
    chat = ChatMessages(tokenizer=str(TOKENIZER), path=chat_parquet(tmp_path))
    prompts = Prompts(tokenizer=str(TOKENIZER), records=("{}",))

    with pytest.raises(ValueError, match="at least one stage"):
        Recipe(model, optim, key, (), str(tmp_path))
    with pytest.raises(ValueError, match="at least one"):
        Stage(name="x", data=chat, steps=0)
    with pytest.raises(ValueError, match="no KL term"):
        Stage(name="x", data=chat, beta=0.1)
    with pytest.raises(ValueError, match="without a reward"):
        Stage(name="x", data=prompts)
