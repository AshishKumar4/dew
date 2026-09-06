# Post-training: SFT, DPO and GRPO on one trainer

Pretraining, SFT, DPO and GRPO are the same loop with different losses. The trainer, the mesh, the EMA, the checkpoints and the logging do not change; what changes is the data path and the objective. Every example below runs from the repository root with the dew venv: the SFT one reads the tiny chat tokenizer under `tests/fixtures/tokenizers`, the DPO and GRPO ones read the parity fixtures under `tests/fixtures/rl`.

## SFT: count the assistant's tokens

SFT data is conversations, and the loss counts only assistant targets. `render_conversation` turns messages into ids plus a role per token, using the tokenizer's own chat template:

```python
from transformers import AutoTokenizer
from dew.data.chat import Role, render_conversation

tokenizer = AutoTokenizer.from_pretrained("tests/fixtures/tokenizers/tiny-chat")
messages = [{"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"}]
ids, roles = render_conversation(tokenizer, messages, "tiny-chat")
targets = (roles[1:] == Role.ASSISTANT).tolist()
assert targets == [False] * 13 + [True] * 16
```

`ChatMessages` packs those roles beside the ids, so a batch carries four aligned per-token fields:

```python
import pyarrow as pa
import pyarrow.parquet as pq
from dew.data import ChatMessages, Loading

pq.write_table(pa.table({"prompt": [messages] * 8}), "sft.parquet")
data = ChatMessages(tokenizer="tests/fixtures/tokenizers/tiny-chat",
                    path="sft.parquet", seq_len=31,
                    loading=Loading(workers=0)).load(batch=8)
batch = next(data.train())
assert {"text", "text_roles", "text_segment_ids", "text_positions"} <= set(batch)
```

The packer adds its own per-field positions beside those four; the loss
reads the four named here.

The objective is the same class as pretraining with one field:

```python
from dew import models
from dew.objectives.lm import LMObjective

decoder = models.build("causal_transformer", vocab_size=len(tokenizer), emb_features=32,
                       num_layers=1, num_heads=2, mlp_features=64, max_seq_len=32)
objective = LMObjective(decoder, 31, loss_role=Role.ASSISTANT)
```
Without `loss_role` every counted target counts, as in pretraining. A batch without `text_roles` raises, naming the column.

## DPO: prefer chosen over rejected

The DPO term is `preference_logsigmoid` in `dew.rl`, pinned against TRL 1.12 on fixed tensors. The fixture carries both sides' per-token log-probabilities, the full-length completion mask, and TRL's own loss:

```python
import jax.numpy as jnp
import numpy as np
from dew.rl import preference_logsigmoid

fixture = dict(np.load("tests/fixtures/rl/dpo.npz", allow_pickle=True))
policy = np.asarray(fixture["policy_logps"], np.float32)
ref = np.asarray(fixture["ref_logps"], np.float32)
mask = np.asarray(fixture["completion_mask"], np.float32)[:, 1:]
half = policy.shape[0] // 2
beta = float(fixture["beta"])
loss = preference_logsigmoid(jnp.asarray(policy[:half]), jnp.asarray(policy[half:]),
                             jnp.asarray(ref[:half]), jnp.asarray(ref[half:]),
                             jnp.asarray(mask[:half]), jnp.asarray(mask[half:]), beta)
assert abs(float(loss) - float(fixture["trl_loss"])) < 1e-6
```

`DPOObjective` composes that term with the chunked head: policy and reference log-probabilities are negated per-token cross entropies, summed under the shifted mask, with the reference read from the frozen `step.ema` at unit decay, so the run carries no second model. Batches are `[B, 2, S]` pairs from `PreferencePairs`, chosen at index 0:

```python
from dew.objectives.rl import DPOObjective

objective = DPOObjective(decoder, 31, beta=0.1)
```

## GRPO: sample, score, clip

The GRPO loss is the `dew.rl` composition pinned against verl 0.9 on one fixed rollout: the dual-clipped surrogate of the token log-ratio, token-meaned over the response mask, plus `beta` times the token-mean k3 KL:

```python
from dew.rl import clipped_surrogate, k3_kl, token_log_ratio, token_mean

fixture = dict(np.load("tests/fixtures/rl/grpo.npz", allow_pickle=True))
get = lambda key: jnp.asarray(np.asarray(fixture[key], np.float32))
old, current, ref, advantages, mask = (get("old_log_probs"), get("current_log_probs"),
    get("ref_log_probs"), get("advantages"), get("response_mask"))
beta = float(fixture["beta"])
ratio = token_log_ratio(current, old)
pg, aux = clipped_surrogate(ratio, advantages, mask)
kl = token_mean(k3_kl(current, ref), mask)
assert abs(float(pg + beta * kl) - float(fixture["verl_loss"])) < 1e-6
```

Online, a `SampledRollout` packs the batch the loss reads: it samples `groups` completions per prompt with `dew.sampling.generate`, scores each with `reward(data_source, completion, ground_truth, extra_info)`, and advantaged with the group or RLOO family. The trainer calls it between the data stream and the compiled step:

```python
from dew import Trainer
from dew.objectives.rl import GRPOObjective, SampledRollout


def reward(data_source, completion, ground_truth, extra_info):
    return float(completion.strip() == ground_truth)


objective = GRPOObjective(decoder, 31, beta=0.01)
rollout = SampledRollout(objective, reward, groups=4, max_new_tokens=32)
trainer = Trainer(objective, optimizer, key=key, rollout=rollout)
```

## Chaining stages

`recipes/chain.py` links stages sharing one decoder. The data names the loss: conversations train SFT, pairs DPO, prompts GRPO. Each stage trains in its own directory, and every stage after the first initializes from the previous stage's final parameters, which also freezes the next stage's reference:

```python
# runs elsewhere: a thousand-step chain over real conversations, pairs and prompts
from recipes.chain import Recipe, Stage

recipe = Recipe(decoder, optimizer, key, stages=(
    Stage(name="sft", data=sft_data, steps=1000),
    Stage(name="dpo", data=dpo_data, steps=500),
    Stage(name="grpo", data=prompt_data, steps=200,
          reward=reward, groups=4, max_new_tokens=32),
), directory="runs/chain", batch=8)
states = recipe.run()
```

`tests/test_chain.py` runs an SFT-to-DPO chain two steps tiny and asserts the link: the second stage's frozen reference is the first stage's final tree, byte for byte. The recipe flag `--objective` in `recipes/lm/train.py` stays `lm | masked_diffusion`: that recipe reads a token directory and a Hugging Face decoder, neither of which a pair or prompt stage survives, so post-training lives in the chain instead of behind its flag.

## Reference parity

| check | reference | largest observed difference |
| --- | --- | --- |
| chat ids and assistant mask | TRL 1.12 | 0, both exact |
| DPO loss | TRL 1.12 | 5.96e-08 |
| DPO gradients | TRL 1.12 autograd | exact |
| GRPO loss | verl 0.9 | 7.45e-08 |
| GRPO gradients | torch autograd | exact |

The parity scripts live in `tools/parity_*.py` and run in an environment with torch and TRL installed; Dew never imports either. The fixtures they write under `tests/fixtures/rl/` are committed, and the tests above read them back.
