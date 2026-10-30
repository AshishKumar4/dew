# Diffusion language models

A diffusion language model writes a span of text by denoising it, all its positions at once, rather than one token after another. Dew trains masked diffusion models (MDLM's objective), runs LLaDA and Dream from their released weights, and runs and fine-tunes DiffusionGemma's block diffusion. The examples run on CPU at toy sizes and show how the calls work; [Language models](language_models.md) covers the autoregressive decoders, tokenizers and token files they share.

The masked example below trains on the same small corpus as that page's, written in the token-store layout:

```python
import jax
import jax.numpy as jnp
import numpy as np
import optax
from pathlib import Path

import dew
from dew import Trainer
from dew.data import ByteTokenizer, presets
from dew.data.sinks import export
from dew.data.sources import TokenFiles
from dew.nn.backbones import CausalTransformer
from dew.optim import OptimConfig

tokenizer = ByteTokenizer()
stories = [
    "Once upon a time, there was a little girl named Lily. She had a red ball.",
    "One day, Lily saw a big dog in the park. The dog wanted to play.",
    "Lily threw the ball and the dog ran after it. They played all day.",
    "At night, Lily went home. She was happy and tired.",
]
ids = np.array(tokenizer.encode(" ".join(stories * 200)), dtype=np.int32)
metadata = {"token_store": True, "tokenizer": "byte", "vocab_size": tokenizer.vocab_size,
            "dtype": "int32", "eos_id": None}
export([{"text": ids[len(ids) // 20:]}], "data/stories-byte/train", format="array_record",
       metadata=metadata, length_field="text")
train = TokenFiles("data/stories-byte/train")
fixtures = Path(dew.__file__).parents[2] / "tests" / "fixtures" / "hf"
```

## Masked diffusion

`MaskedDiffusionObjective` trains a bidirectional decoder to recover masked tokens under MDLM's negative ELBO. As in MDLM's SUBS parameterization, the mask token gets no probability mass, so its column is left out of the cross entropy's partition. LLaDA's published training script takes the cross entropy over the whole vocabulary, so its loss values differ from Dew's by that term. Each row holds exactly `seq_len` tokens, because there is no next-token shift; `token_windows` at `seq_len=63`, whose rows hold 64 IDs, feeds a 64-token objective. The mask token needs an ID of its own, here the one after the last byte:

```python
from dew.diffusion.discrete import MDLM
from dew.objectives.diffusion import MaskedDiffusionObjective

masked_data = train.shuffle(seed=0).repeat().pipe(presets.token_windows, seq_len=63, batch=16)
mask_id = tokenizer.vocab_size
process = MDLM(mask_id=mask_id)()
masked_model = CausalTransformer(vocab_size=mask_id + 1,
                                 emb_features=64, num_layers=2, num_heads=2,
                                 mlp_features=256, max_seq_len=64, causal=False)
masked_objective = MaskedDiffusionObjective(masked_model, process, seq_len=64)
masked_state = Trainer(masked_objective, OptimConfig(learning_rate=3e-3), key=jax.random.key(4)).fit(
    masked_data, steps=300, log_every=150)
drawn = process.generate(masked_model, masked_state.variables,
                         [tokenizer.encode("One day, Lily")], 24, key=jax.random.key(5))
print(tokenizer.decode(drawn.tokens[0]))
```

```text
Training CausalTransformer from step 0 to 300: 147,968 parameters, on 1 × cpu, batch 16, float32
step 150/300  loss 2.317  masked_accuracy 32.2%  masked_fraction 47.3%  step_time_ms 24.75  samples_per_sec 646.4  accepted 100.0%  0:00:04 left
step 300/300  loss 1.414  masked_accuracy 43.1%  masked_fraction 50.5%  step_time_ms 21.53  samples_per_sec 743.0  accepted 100.0%
Trained 300 steps in 0:00:16: first step after 9.75 s, then 44.8 step/s
40.6% of the wall time in steps, final loss 1.414
One day, Lily tond �ara d Linted thed
```

The loss is MDLM's negative ELBO per token. `process.generate` unmasks the 24 tokens after the prompt together, in MDLM's 128 reveal steps by default (`steps=`), then fills what is still masked with the model's most likely tokens, as MDLM's noise removal does. A model this small, after 300 steps, draws broken fragments of words rather than a sentence, and a byte it draws that ends no character decodes as `�`.

On TinyStories, with the GPT-2 tokenizer, a 4-layer, 256-wide model trained on 128-token rows at batch 64 took about five minutes for 4,000 steps on an L4. Its loss fell from 3.48 at step 1,000 to 2.78 at step 4,000, and validation perplexity reached 15.0. That perplexity is exp of the ELBO, an upper bound on the model's own perplexity, so it does not compare directly with an autoregressive decoder's. One draw of 48 tokens read:

> Once upon a time, in a small town, there lived a little girl named Lily. Mia loved whistle. She had an key telling to try to sleep. One day, she always to see her favorite mom, dad unate to have lots of.

Dew runs LLaDA and Dream through this masked-token path, starting from their released weights. They predict masked tokens with bidirectional attention and need a mask token ID and a masked-diffusion objective. Replacing the autoregressive loss alone is not enough, because the attention and the corruption process change too. For these models, `PretrainedMaskedDecoder.load(...).text_generation()`, `dew.pipeline(source_or_run)` and `MaskedDiffusionObjective.pipeline(state)` return `MaskedGeneration`. It runs Dew's native MDLM algorithm (`DiscreteProcess` with `Unmask`) for LLaDA and Dream, and does not run their own remasking recipes. LLaDA2.0 generates as its release does, and LLaDA2.1 by the same walk without its release's editing of revealed tokens: its task walks the response 32 places at a time over 32 steps each, each position reading its document through its own block and no later one, and reveals with `Confident`, the draws past 0.95 of their distribution, else each step's quota of the most confident (`DiscreteProcess.generate(..., block=)`). At temperature 0 `Confident` takes each position's best token, where the release's `generate` documents greedy decoding but still draws, and Dew's walk never draws the mask token, which the release's can. `MaskedSettings`, `MaskedGeneration` and `MaskedDiffusionObjective` take `Confident` and `block` for any masked model, an objective over a LLaDA2.0 source taking its walk by default; the objective's previews still walk whole rows. LLaDA2.2 generates with an editing stage Dew does not run, so its task refines the whole response with MDLM.

```python
from dew.interop import PretrainedMaskedDecoder

llada = PretrainedMaskedDecoder.load(fixtures / "llada-tiny", dtype=jnp.float32, attention_impl="xla")
masked_task = llada.text_generation()
result = masked_task([[1, 2, 3]], 8, key=7, steps=16, n=2)
print(result.host().tokens)
```

```text
[[  1   2   3  34  71  40  47  69  66  55  36]
 [  1   2   3  88 125  84  65  97 111   3 112]]
```

Released sources that ship a tokenizer also accept text and provide `result.text`. This path handles text only and rejects media payloads and media token fields; numeric requests use `ModelInputs`. Attention validity, logical `[B, S]` positions, rotary coordinates and segment IDs pass through the masked model. The prompt stays fixed, even if it contains a literal mask ID. An MDLM walk refines all requested response positions together under bidirectional attention, a blockwise walk one block at a time, each position reading its document through its own block; mask IDs are never drawn as generated tokens.

The default is 64 model evaluations, including the final clean prediction; a call can override `steps`. The `n` continuations are grouped by prompt, and continuation zero does not change when you ask for more. EOS is applied after the whole span is refined, so lengths include the first EOS and the rest is padded; an MDLM walk does not stop early at an EOS, while LLaDA2's blockwise walk stops a row after the block that revealed its end, its refinement count the steps its blocks took. A request for zero tokens keeps the prompt and reports zero refinements. `CanvasGeneration` reports lengths, EOS flags and refinement counts, and has no autoregressive log-probabilities. This path does not accept autoregressive sampling, beam or logits controls.

A source generation control that native MDLM cannot follow is rejected by name. Neutral values are accepted, as are the shared budget, continuation count, EOS and padding metadata. MDLM does not use a KV cache, so `use_cache=False` is accepted and asking for a cache is not.

Saved masked-diffusion recipe runs keep their compute and storage precision, and `dew.pipeline`'s `ema` option picks the live or EMA weights. The run record rebuilds native MDLM with its default `Unmask` solver and 128 steps, MDLM's own, so custom objective steps or solver choices set in code are not saved. Plain `Checkpoints` saves weights and training state but not these task settings, so you have to apply a custom task configuration again. The run saves its end-of-sequence ids: an objective over a loaded source takes the source's (`Pretrained.eos_token_ids`), and `eos_token_ids=` sets them otherwise, so the run's task and the task its record rebuilds stop a response there, as the source's own task does.

## DiffusionGemma

DiffusionGemma uses uniform-vocabulary corruption, a causal prompt encoder, a bidirectional canvas and self-conditioning. `Pretrained.load` loads the complete model. Its `BlockProcess` generation policy refines whole canvases and commits clean tokens to the shared text cache. It returns `CanvasGeneration` with response lengths (including EOS), termination flags and per-row refinement counts, and no autoregressive likelihood fields.

The next example loads the tiny reference checkpoint, then tokenizes, generates, decodes, saves and reloads it. Its synthetic vocabulary has 64 tokens. The released model ID is `google/diffusiongemma-26B-A4B-it`, which downloads large weights and needs matching host and device memory.

```python
from tempfile import TemporaryDirectory

from dew.interop import PretrainedBlockDecoder

gemma = PretrainedBlockDecoder.load(fixtures / "diffusion-gemma-workflow", dtype=jnp.float32,
                                    attention_impl="xla", max_seq_len=32)
prompt = gemma.processor(["<bos> t5 t7 t9 t11"])
block_task = gemma.block_generation()
generated = block_task(prompt, 7, key=jax.random.key(11))
print(block_task.decode(generated))
with TemporaryDirectory() as checkpoint:
    gemma.save(checkpoint)
    restored = PretrainedBlockDecoder.load(checkpoint, dtype=jnp.float32, attention_impl="xla",
                                           max_seq_len=32)
    replay = restored.block_generation()(prompt, 7, key=jax.random.key(11))
    np.testing.assert_array_equal(replay.tokens, generated.tokens)
```

```text
('t37 t49 t62 t14 t23 t49 t34',)
```

The last canvas is refined at full width, and the returned response is then cut to `max_new_tokens`. The prefix plus the canvas capacity, rounded up to whole canvases, must fit in `max_seq_len`. Generation defaults come from `generation_config.json`; pass a `BlockProcess` as `process=` to override them. Media go through the checkpoint's processor and run only during prompt prefill, not on every refinement. The [training-contract note](../research/inference.md#diffusiongemma-training-contract-and-open-prerequisites) explains which part of training is public (the official fine-tuning recipe) and which Google has not published (the original sampler-distillation and RL objective).

### Block diffusion fine-tuning

`BlockDiffusionObjective` ports Google's public SFT adapter, released after the model. It samples a valid response canvas, corrupts the whole response with uniform-vocabulary noise, and runs self-conditioning with a detached first pass. It then combines the canvas loss and the encoder loss, each normalized per row on its own. The default time safety margin is 1e-4 and the self-conditioning probability is 0.5. It is not the unpublished sampler-distillation and RL objective.

The next example takes one optimizer step on the tiny reference model, with a synthetic vocabulary and unequal target support. This dense model has neither the routed experts nor the vision tower that transformers' `DiffusionGemmaForBlockDiffusion` builds, so `save` refuses to export it, and you keep it as a run checkpoint instead.

A run of the image-reading model exports with `Pretrained.from_run(run).save(destination)`, which writes the config from the run's model: the text stack, the Gemma 4 tower's `vision_config` and the canvas length. Dew places images by position and keeps no image token IDs, so transformers' defaults apply to them. Because transformers reads `generation_config.json` as a closed set of fields, the run must use a Hugging Face tokenizer, whose files are its whole record, and a run on Dew's byte vocabulary is refused.

```python
from dew.objectives.diffusion import BlockDiffusionObjective
from dew.data.dataset import Loader

sft_source = fixtures / "diffusion-gemma-sft"
sft_bundle = PretrainedBlockDecoder.load(sft_source, dtype=jnp.float32, attention_impl="xla", max_seq_len=32)
with np.load(sft_source / "reference.npz") as reference:
    train_tokens = np.tile(reference["tokens"], (jax.device_count(), 1))
block_data = Loader(train=lambda partition: iter([{"text": train_tokens}]), val=None,
                     records=len(train_tokens), batch=len(train_tokens))
block_objective = BlockDiffusionObjective(sft_bundle, prompt_length=4, num_canvases=2)
block_state = Trainer(block_objective, optax.sgd(0.001), key=jax.random.key(2)).fit(
    block_data, steps=1, log_every=1)
print("Optimizer updates:", int(block_state.updates))
```

```text
Training DiffusionGemma from step 0 to 1: 1,858 parameters, on 1 × cpu, batch 2, float32
step 1/1  loss 6.674  canvas_ce 3.131  encoder_ce 3.542  step_time_ms 93.39  samples_per_sec 21.42  accepted 100.0%
Trained 1 steps in 0:00:15: first step after 15.06 s
0.1% of the wall time in steps, final loss 6.802
Optimizer updates: 1
```

`recipes/lm/train.py --objective block_diffusion` runs the same objective; there is no separate recipe. It trains on complete token-window rows, and the chain's `seq_len + 1` must equal `--objective.prompt-length` plus a whole number of training canvases. `--objective.canvas-size` defaults to the checkpoint's canvas length. Packed documents are rejected, because their context boundaries differ. Block SFT logs `canvas_ce` and `encoder_ce`. It does not report autoregressive perplexity or use the autoregressive preview settings.

```bash
python recipes/lm/train.py --set data.source.url=data/diffusion-token-windows \
    --set data.token_windows.seq_len=511 --objective block_diffusion --objective.prompt-length 256 \
    --pretrained google/diffusiongemma-26B-A4B-it \
    --tokenizer google/diffusiongemma-26B-A4B-it \
    --max-new-tokens 0 --optim.learning-rate 0.00015
```

The token files must use the checkpoint's tokenizer and hold the clean prompt prefix followed by the response canvases you intend. Trainer checkpoints keep the optimizer and iterator state, and `Pretrained.save` writes a complete inference checkpoint in the source format.
