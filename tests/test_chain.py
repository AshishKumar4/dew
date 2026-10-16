"""A recipe chain trains every stage kind on one decoder and links their checkpoints.

A `Recipe` runs its stages in order, each in its own directory, with every
stage after the first initializing from the model's part of the previous
stage's final variables. The observable link: a DPO, GRPO or PPO stage's
frozen reference, written at its first step, is the tree the previous stage
handed on, leaf for leaf. Pretraining and a distilled SFT stage hand on the
decoder alone, without the teacher; PPO hands on its policy, without the
critic. A GRPO stage's reward reads text its prompts' tokenizer decodes,
and sampling stops at that tokenizer's EOS. A stage that names the wrong
data or settings fails the run before its first stage trains.
"""

import dataclasses
import functools
import json
from pathlib import Path

import jax
import numpy as np
import optax
import pytest
from bank_support import identical
from episode_support import GROUPS, PROMPT, RESPONSE, SAMPLING, TURNS, Harness, ToolPolicy, verify
from flax import linen as nn
from rl_support import TinyHead

from dew.config.data import ChatData, PreferenceData, PromptData, RecordsData, TextData
from dew.data.text import tokenizer_for
from dew.objectives.base import part
from dew.objectives.distillation import TEACHER
from dew.objectives.lm import LMObjective
from dew.objectives.rl import ValueHead
from dew.training import Layout
from recipes.chain import Episodes, Recipe, Stage, Teacher

TOKENIZER = Path(__file__).resolve().parent / "fixtures" / "tokenizers" / "tiny-chat"
LAYOUT = Layout(min_shard=1, tolerance=1.0)


@dataclasses.dataclass(frozen=True)
class Loaded:
    """A loaded model (`dew.objectives.base.Source`): a module and its tree."""

    model: nn.Module
    variables: dict
    text_processor: None = None


def moved(after, before) -> bool:
    """Whether training changed a leaf of a tree that kept its structure."""
    assert jax.tree.structure(after) == jax.tree.structure(before)
    return any(not np.array_equal(new, old)
               for new, old in zip(jax.tree.leaves(after), jax.tree.leaves(before), strict=True))


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


def pairs():
    return PreferenceData(source=tuple(json.loads(row) for row in pair_records()), seq_len=4)


def chat(tmp_path):
    return ChatData(source=chat_parquet(tmp_path), tokenizer=str(TOKENIZER), seq_len=31, long="chunk")


def stages(tmp_path):
    return (Stage(name="sft", data=chat(tmp_path), steps=1), Stage(name="dpo", data=pairs(), steps=1))


def test_each_stage_continues_the_last(tmp_path):
    """Two one-step stages: both checkpoint directories land, the second
    stage trains (its parameters move off the first stage's), and its frozen
    reference is the first stage's final tree."""
    recipe = Recipe(TinyHead(vocab_size=8), optax.sgd(0.1), jax.random.key(0),
                    stages(tmp_path), str(tmp_path / "chain"), batch=8, layout=LAYOUT)

    states = recipe.run()

    assert len(states) == 2
    assert (tmp_path / "chain" / "sft").is_dir()
    assert (tmp_path / "chain" / "dpo").is_dir()
    identical(states[1].ema, states[0].variables)
    assert moved(states[1].variables, states[0].variables)


def test_pretraining_and_distillation_hand_on_the_decoder_alone(tmp_path):
    """Pretraining on text, DPO, SFT distilled from a frozen teacher, then
    DPO again. Each DPO stage's reference is the final decoder tree of the
    stage before it; the distilled stage trains the student, leaves the
    teacher as it came in, and hands on the student without it."""
    vocab = tokenizer_for(str(TOKENIZER)).vocab_size
    held = LMObjective(TinyHead(vocab_size=vocab), 31).init(jax.random.key(7))
    teacher = Teacher(Loaded(TinyHead(vocab_size=vocab), held), alpha=0.5, temperature=2.0)
    text = TextData(source=tuple({"text": "the quick brown fox jumps over the lazy dog"} for _ in range(16)),
                    tokenizer=str(TOKENIZER), seq_len=8, token_holdout=None)
    recipe = Recipe(TinyHead(vocab_size=vocab), optax.sgd(0.1), jax.random.key(0), (
        Stage(name="pretraining", data=text, steps=1),
        Stage(name="dpo", data=pairs(), steps=1),
        Stage(name="sft", data=chat(tmp_path), steps=1, teacher=teacher),
        Stage(name="aligned", data=pairs(), steps=1),
    ), str(tmp_path / "chain"), batch=8, layout=LAYOUT)

    states = recipe.run()

    identical(states[1].ema, states[0].variables)
    distilled = states[2].variables
    identical(distilled[TEACHER], held)
    student = {name: tree for name, tree in distilled.items() if name != TEACHER}
    assert moved(student, states[1].variables)
    identical(states[3].ema, student)


def test_a_ppo_stage_acts_in_its_environment_and_hands_on_its_policy(tmp_path):
    """DPO, PPO over tool episodes, then DPO on one bigram tool policy. The
    PPO stage opens its episodes in the environment and keeps the DPO
    stage's final tree as its reference, and the last stage's reference is
    the PPO policy without the critic."""
    harness = Harness()
    episodes = Episodes(harness, verify, ValueHead(ToolPolicy()), max_prompt_tokens=PROMPT,
                        max_turns=TURNS, gamma=.97, lam=.9)
    tasks = RecordsData(source=tuple({"task_id": index} for index in range(8)))
    recipe = Recipe(ToolPolicy(), optax.sgd(.05), jax.random.key(19), (
        Stage(name="dpo", data=pairs(), steps=1),
        Stage(name="ppo", data=tasks, steps=1, beta=.03, groups=GROUPS, max_new_tokens=RESPONSE,
              sampling=SAMPLING, episodes=episodes),
        Stage(name="aligned", data=pairs(), steps=1),
    ), str(tmp_path / "chain"), batch=8, layout=LAYOUT)

    states = recipe.run()

    assert len(harness.opened) == 8 * GROUPS and harness.calls
    identical(part(states[1].ema, "policy"), states[0].variables)
    policy = part(states[1].variables, "policy")
    assert moved(policy, states[0].variables)
    identical(states[2].ema, policy)


def leaning(vocab: int, letter: int, eos: int) -> np.ndarray:
    """Bigram logits that draw `letter` about three times in four after any other id, `eos` the
    rest of the time, and always `eos` after `letter`."""
    logits = np.full((vocab, vocab), -1000., np.float32)
    logits[:, letter], logits[:, eos] = 1., 0.
    logits[letter, letter], logits[letter, eos] = -1000., 1000.
    return logits


def test_a_grpo_stage_scores_decoded_text_and_stops_at_the_tokenizer_eos(tmp_path):
    """DPO, GRPO, then DPO on a bigram policy that draws "a" then EOS, or
    EOS at once. Of the 8 * 4 completions the reward reads, each is "a" or
    empty as the prompts' tokenizer decodes it, and both occur: EOS stopped
    every one inside the four-token budget, where sampling past it would
    have decoded EOS text or a second "a". GRPO trains on the differing
    rewards, its reference is the first stage's final tree, and the last
    stage's reference is GRPO's."""
    tokenizer = tokenizer_for(str(TOKENIZER))
    letter, = tokenizer.encode("a", add_special_tokens=False)
    assert tokenizer.eos_id is not None
    model = ToolPolicy(vocab_size=tokenizer.vocab_size, max_seq_len=12,
                       logits=functools.partial(leaning, tokenizer.vocab_size, letter, tokenizer.eos_id))
    read = []

    def reward(source, text, truth, info):
        read.append(text)
        return float(len(text))

    prompts = PromptData(source=tuple({"prompt": "hi"} for _ in range(8)), tokenizer=str(TOKENIZER),
                         max_prompt_len=8, long="truncate")
    states = Recipe(model, optax.sgd(.1), jax.random.key(0), (
        Stage(name="dpo", data=pairs(), steps=1),
        Stage(name="grpo", data=prompts, steps=1, beta=.01, reward=reward, max_new_tokens=4),
        Stage(name="aligned", data=pairs(), steps=1),
    ), str(tmp_path / "chain"), batch=8, layout=LAYOUT).run()

    assert len(read) == 8 * 4
    assert set(read) == {"", "a"}
    identical(states[1].ema, states[0].variables)
    assert moved(states[1].variables, states[0].variables)
    identical(states[2].ema, states[1].variables)


def test_a_mislinked_chain_is_refused_before_anything_trains(tmp_path):
    """Each refused stage follows a sound one, and the run fails before the
    sound stage writes its directory."""
    model, optim, key = TinyHead(vocab_size=8), optax.sgd(0.1), jax.random.key(0)
    text = TextData(source=({"text": "hi"},), tokenizer=str(TOKENIZER))
    prompts = PromptData(source=tuple(json.loads(row) for row in ("{}",)), tokenizer=str(TOKENIZER),
        long="truncate")
    tasks = RecordsData(source=({"task_id": 0},))
    episodes = Episodes(Harness(), verify, ValueHead(ToolPolicy()), max_prompt_tokens=PROMPT,
                        max_turns=TURNS, gamma=1., lam=1.)
    teacher = Teacher(Loaded(model, {}), alpha=0.5, temperature=1.0)

    with pytest.raises(ValueError):
        Recipe(model, optim, key, (), str(tmp_path))
    with pytest.raises(ValueError):
        Stage(name="x", data=chat(tmp_path), steps=0)
    with pytest.raises(TypeError):
        Teacher(model, alpha=0.5, temperature=1.0)
    refused = (
        Stage(name="x", data=chat(tmp_path), beta=0.1),
        Stage(name="x", data=text, beta=0.1),
        Stage(name="x", data=prompts),
        Stage(name="x", data=tasks),
        Stage(name="x", data=pairs(), episodes=episodes),
        Stage(name="x", data=pairs(), teacher=teacher),
        # PPO actions end at an EOS, and this stage's sampling names none.
        Stage(name="x", data=tasks, max_new_tokens=RESPONSE, episodes=episodes),
    )
    for stage in refused:
        with pytest.raises(ValueError):
            Recipe(ToolPolicy(), optim, key, (Stage(name="dpo", data=pairs()), stage),
                   str(tmp_path / "chain")).run()
    assert not (tmp_path / "chain").exists()
