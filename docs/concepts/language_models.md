# Language models

An autoregressive language model is another objective on the same trainer. The mesh, the sharding, the EMA, the checkpoints and the logging are the ones diffusion and JEPA runs use; what differs is the data (token ids instead of images), the loss (cross entropy against the next token) and the validation artifact (text the model wrote).

## From text to batches

`tools/tokenize_text.py` turns a file or a directory of files into a token directory:

```bash
python tools/tokenize_text.py --input data/shakespeare.txt \
    --out data/shakespeare-byte --tokenizer byte --val-fraction 0.01
```

- `train.bin` and `val.bin` are flat arrays of token ids in the smallest unsigned dtype that holds the vocabulary (`uint8` up to 256 ids, `uint16` up to 65536, `uint32` beyond), with no separators or padding.
- `meta.json` records the `tokenizer`, its `vocab_size`, the `dtype` of the arrays and the token counts. The recipe reads the vocabulary from here, so the model is always built for the ids on disk.

`--tokenizer` is `byte` (`dew.data.ByteTokenizer`, raw UTF-8 bytes, vocabulary 256) or the name of a Hugging Face tokenizer (`dew.data.HFTokenizer`). Both have `encode(str) -> list[int]` and `decode(ids) -> str`.

`TokenWindows(path, seq_len).load(batch=...)` reads windows of `seq_len + 1` ids off the memory-mapped file, so a batch is `{"text": int32[B, seq_len + 1]}` and carries its own labels: inputs are `text[:, :-1]`, targets `text[:, 1:]`.

## Packed documents

A flat stream has no document boundaries, so a fixed window can straddle two unrelated texts. `--pack` closes every input file with the tokenizer's eos id and records it in `meta.json`:

```bash
python tools/tokenize_text.py --input data/corpus --out data/corpus-byte --tokenizer byte --pack
```

`PackedTokens` fills each window with whole documents (grain's first-fit packer; documents longer than the window are cut into consecutive pieces first) and yields

```text
{"text": int32[B, seq_len + 1],
 "text_segment_ids": int32[B, seq_len + 1],   # which document, 0 for padding
 "text_positions": int32[B, seq_len + 1]}     # position inside that document
```

The backbone takes `positions` and `segment_ids`: RoPE rotates by the position inside the document and the attention mask is causal and block-diagonal, so no query reaches another document or the padding. The objective drops the one target a packed row must not train on, the last token of a document predicting the first of the next, along with padding.

A segment mask costs the fused kernel. cuDNN takes no mask argument, so a packed batch runs on the XLA kernel, which materialises the fp32 attention logits per layer. On an RTX 4080 with three layers of GPT-2 small's width in bf16:

| shape | batch | kernel | ms/step | peak GiB |
|---|---|---|---:|---:|
| 16 x 512 | fixed window | cuDNN | 75.8 | 4.99 |
| 16 x 512 | packed, 4 documents | xla | 83.6 | 5.80 |
| 4 x 2048 | fixed window | cuDNN | 78.5 | 5.00 |
| 4 x 2048 | packed, 4 documents | xla | 108.2 | 8.34 |

```bash
python tools/benchmark_step.py --preset small --architectures causal_transformer --steps 20
python tools/benchmark_step.py --preset small --architectures causal_transformer --steps 20 --packed-documents 4
```

Sharding happens before packing (each process slices the documents), and the packed iterator's position saves and restores with the checkpoint like the fixed-window one.

## The objective

```python
from dew.objectives.lm import LMObjective, Samples

objective = LMObjective(model, seq_len=256, samples=Samples(
    prompt=tokenizer.encode("To be, or not to be"),
    max_new_tokens=128, temperature=0.8, decode=tokenizer.decode))
```

`loss` multiplies the final hidden states by the head matrix one vocabulary slice at a time and returns the mean cross entropy in float32, without building the full `[tokens, vocab]` logits tensor; `head_chunks` (default 4) is how many slices. A bfloat16 logsumexp over a large vocabulary moves the loss and the gradient, so the head runs in float32. The model exposes `hidden_states(...)` and `head_weight(params)` for this; `CausalTransformer` does.

`pad_id` excludes padded targets from the mean and defaults to `None`, because a fixed-window token file has no padding. A packed batch needs no pad id; its segment ids say which slots are padding.

`mtp_weight` turns on DeepSeek's multi-token-prediction term for a model built with `num_nextn_predict_layers` above zero (arXiv 2412.19437, section 2.2). Depth d of the model pairs the previous depth's state at each position with the embedding of the token d further on and scores the token after that through the shared head; the training loss adds `mtp_weight` times the mean over the depths of each depth's cross entropy, normalised by the same target count as the main term and with a packed batch's document boundaries weighted out the same way. `train/mtp_ce` reports the mean depth cross entropy. Unset, the loss is the main cross entropy alone and the depths get no gradient.

`ema` averages the whole parameter tree at `ema_decay`, and the EMA copy is what validation reads. At validation the objective returns `TokenScores` for `metrics.perplexity()`, which reduces to the exponential of the target-weighted mean over the whole pass, and `TextSamples` when `samples` are configured.

## Generation

```python
import jax, jax.numpy as jnp
from dew import models
from dew.sampling import generate

lm = models.CausalTransformer(vocab_size=256, emb_features=32, num_layers=1, num_heads=2, max_seq_len=16)
variables = lm.init(jax.random.key(0), jnp.zeros((1, 4), jnp.int32))
tokens = generate(lm, variables, jnp.array([list(b"ROME")]), max_new_tokens=8,
                  key=jax.random.key(1), temperature=0.8, top_k=40)
assert tokens.shape == (1, 12)
```

`generate` prefills the KV cache on the prompt and runs one `lax.scan` over the decode steps, returning `int32[B, P + max_new_tokens]`. `temperature=0` is greedy and `top_k` truncates the distribution before sampling. The second argument is the variables dict the trainer holds, so the EMA copy samples directly.

## Pretrained checkpoints

`dew.interop.hf_decoders` loads a Hugging Face decoder into the same `CausalTransformer` a from-scratch run trains:

```python
# runs elsewhere: downloads a Qwen checkpoint from the Hub
from dew.interop import load_pretrained_decoder, save_pretrained_decoder

model, variables, config = load_pretrained_decoder("Qwen/Qwen3-0.6B")
logits = model.apply(variables, tokens)              # [B, S, 151936] fp32
save_pretrained_decoder(model, variables, "out/qwen3-tuned", tokenizer_name="Qwen/Qwen3-0.6B")
```

`load_pretrained_decoder(name_or_dir, *, dtype="bfloat16", attention_impl="auto", max_seq_len=None, revision=None)` takes a hub repo id or a local directory. It downloads the safetensors and JSON only, reads the weights without torch, and builds the model through `models.build`; `dtype` is the compute dtype and the parameters stay float32. The third return is the dew config the model was built from, the one a run logs.

A checkpoint quantized the way DeepSeek V3 and V3.2 ship, every linear an F8_E4M3 weight with an F32 `weight_scale_inv` of one scale per 128 x 128 block (the config's `quantization_config` names `fp8`, `e4m3` and `weight_block_size [128, 128]`), loads dequantized: the reader widens each fp8 weight to float32 exactly and `dew.interop.dequantize_checkpoint` multiplies it by its block scales, the formula of DeepSeek's `weight_dequant` (inference/kernel.py), one float32 multiply per element with the partial last block a shape like `[576, 7168]` leaves. `dequantize_fp8_blocks(weight, scale_inv, block)` is that arithmetic on one tensor, and tests/test_quantized.py holds it to the reference bit for bit, on random blocks and on the real `kv_a_proj_with_mqa` pair of DeepSeek-V3 (network-marked). A checkpoint quantized another way (GPT-OSS's MXFP4) is its own reader's business and passes this one untouched.

`translate_config` and `translate_weights` are the two halves on their own: a Linear's `.weight` becomes a transposed `.kernel`, a norm's `.weight` becomes `.scale`, and a tied `lm_head.weight` is dropped after a check that it is the embedding's copy. `save_pretrained_decoder` writes `config.json`, `model.safetensors` and `generation_config.json` back in the family's vocabulary, and transformers loads the result.

| Family | `model_type` | Loads |
|---|---|---|
| Llama 2, 3, 3.1 | `llama` | dense GQA decoders; 3.1's `rope_scaling` on the rotary table |
| Mistral | `mistral` | Llama with a sliding window on every layer |
| Mixtral | `mixtral` | the routed mixture with the softmax router |
| Qwen 2 | `qwen2` | q, k and v biases over a bias-free `o_proj` |
| Qwen 3 | `qwen3` | dense; q/k norms, sliding layers |
| Qwen3-MoE | `qwen3_moe` | the mixture with `norm_topk_prob`, `decoder_sparse_step` and `mlp_only_layers` |
| Gemma 1, 2 | `gemma`, `gemma2` | exact GeGLU; Gemma 2 adds the attention logit softcap (runs on the xla kernel), alternating windows and post norms |
| OLMo 3 | `olmo3` | post-norm block, q/k norms over the whole projection; the released config's full-layer YaRN raises a `ValueError` |
| Gemma 3 | `gemma3_text` | `gemma-3-1b-pt`; the larger sizes are multimodal repos with a linear RoPE factor and raise |
| Gemma 3n | `gemma3n_text` | E2B and E4B: AltUp's copies of the residual stream (`altup_num_inputs`, `altup_active_idx`, `altup_coef_clip`, `altup_correct_scale`), the LAuReL block (`laurel_rank`), gaussian top-k activation sparsity (`activation_sparsity_pattern`), one feed-forward width per layer, per-layer inputs and KV sharing; the released repos are multimodal wrappers whose `text_config` alone translates |
| Gemma 4 | `gemma4_text` | the text decoder of every size: per-layer inputs, KV sharing, partial rotary, logit softcap, the global layers' own head dim and key/value count, and for the 26B-A4B the routed experts summed beside each layer's dense MLP (`enable_moe_block`), the global layers reading their values off the keys (`attention_k_eq_v`) and the per-layer output scalar |
| Qwen 3.5 | `qwen3_5_text` | the hybrid of gated delta net layers and gated full-attention layers; the released repos are multimodal wrappers and raise |
| GPT OSS | `gpt_oss` | alternating sliding and full layers with a learned attention sink per head, YaRN over grouped-query heads, the biased router and the clamped interleaved experts with their bias vectors; MXFP4 blocks and scales unpack to bf16 on load (`dew.nn.gpt_oss.dequantize_mxfp4`) |
| DeepSeek V2, V2-Lite | `deepseek_v2`, `deepseek_v2_lite` | MLA with V2's dims and no indexer, the softmax router under `greedy` or `group_limited_greedy` without renormalisation, the expert-level balance loss (`aux_loss_alpha`, `seq_aux`) on `LMObjective` |
| DeepSeek V3, V3.2 | `deepseek_v3`, `deepseek_v32` | multi-head latent attention with YaRN, the V3.2 sparse indexer, the sigmoid router with grouping, scaling and the balancing bias, shared experts; `num_nextn_predict_layers` reads as 0 because the released repos ship no MTP weights |
| Kimi K2 | `kimi_k2` | DeepSeek V3's computation under its own model_type and tokenizer; what it computes exports as `deepseek_v3` |
| GLM 4.5, GLM 5 | `glm4_moe` | biased q/k/v over a bias-free o_proj, a half rotary, DeepSeek V3's router with the shared experts and dense first layers, and the MTP depths the checkpoint ships (`num_nextn_predict_layers`), each composed as the serving engines run it |
| Llama 4 (text) | `llama4_text` | iRoPE: chunked rotated local layers around global layers with no rope and temperature scaling, every interleaved layer routed with a shared expert and the routing weight on the expert input; the released repos are multimodal wrappers whose `text_config` alone translates |
| LLaDA | `llada` | fully bidirectional Llama block with a reserved mask id (`mask_token_id`); the OLMo-style tensor names remap onto the llama layout with no second table |
| Dream | `dream`, `Dream` | Qwen2.5-shaped decoder with hard-coded full attention and a mask id, on the qwen2 tensor layout |
| DiffusionGemma (text) | `diffusion_gemma_text` | the Gemma 4 text weights in decoder mode; the block-diffusion sampler around them lives in `dew.diffusion.block` |

A config field that changes what the model computes and has no counterpart here raises a `ValueError` naming the field (`use_bidirectional_attention` other than Gemma 4's vision-only spelling, a `mlp_bias`, an activation other than silu or tanh-gelu, a `rope_scaling` the family's reference does not read, DeepSeek V2's `norm_topk_prob` or a `scoring_func` other than softmax, GPT OSS's `router_jitter_noise` or a quantization other than MXFP4, Llama 4's `layer_types` disagreeing with `no_rope_layers`).

A multimodal gemma3, llama4, gemma4 or qwen3_5 repo translates through `translate_wrapper_config` into its decoder, tower and projector records (see below); a gemma3n wrapper is refused on its MobileNet-v5 tower, and its `text_config` is what translates on its own. The families still to come are in the README roadmap.
Every family lands with a parity test: `tools/hf_reference.py` writes fixtures under torch and transformers, and `tests/test_hf_decoders.py` compares logits at float32 with the tolerance and the largest observed difference written in the test. Qwen3-0.6B's real weights agree with the reference on the argmax at every position.

`recipes/lm/train.py --pretrained` continues training one of these:

```bash
python tools/tokenize_text.py --input data/corpus.txt --out data/corpus-qwen3 --tokenizer Qwen/Qwen3-0.6B
python recipes/lm/train.py data:token-windows --data.path data/corpus-qwen3 --data.seq-len 512 \
    --pretrained Qwen/Qwen3-0.6B --tokenizer Qwen/Qwen3-0.6B --trainer.batch-size 4 \
    --optim.learning-rate 1e-5
```

The checkpoint decides every architecture field, and the token files have to come from the checkpoint's own tokenizer; a `meta.json` written with a different one stops the run.

## Masked diffusion decoders

LLaDA and Dream train masked diffusion. Each row is corrupted by masking a
fraction of its positions with the reserved mask id, and the loss is the cross
entropy on the masked positions weighted by the process. The backbone is the
same `CausalTransformer` with `causal=False`, and the objective is
`masked_diffusion` in `dew.objectives.diffusion.masked` over the `mdlm`
process in `dew.diffusion.discrete`.

```python
# runs elsewhere: downloads a LLaDA checkpoint from the Hub
from dew.interop import load_pretrained_decoder

model, variables, config = load_pretrained_decoder("GSAI-ML/LLaDA-8B-Base")
```

`recipes/lm/train.py --objective masked_diffusion --pretrained` with a diffusion
checkpoint trains one. The checkpoint carries its mask id, so the two flags are
the whole command.

```bash
python recipes/lm/train.py data:token-windows --data.path data/corpus-llada --data.seq-len 512 \
    --pretrained GSAI-ML/LLaDA-8B-Base --objective masked_diffusion --tokenizer GSAI-ML/LLaDA-8B-Base
```

DiffusionGemma denoises blocks. A uniform random canvas of `canvas_length` ids
sharpens over `max_steps` steps under a temperature annealed from `t_max` to
`t_min`, positions whose categorical entropy fits `entropy_bound` are accepted
and the rest are redrawn, and the previous step's tempered logits condition the
next one through the self-conditioning MLP in `dew.nn.diffusion_gemma`.
`BlockProcess` in `dew.diffusion.block` holds the schedule and `sample_canvas`
runs the loop over a denoiser callable. The wired denoiser is two functions
beside it: `prefill_cache` runs the causal encoder over the prompt into a KV
cache, and `denoise_logits` runs the canvas against that cache bidirectionally
with self-conditioning folded in through the decoder's input-embeddings hook.

```python
from dew.diffusion.block import BlockProcess

process = BlockProcess(canvas_length=256, vocab_size=262144)
process.temperature(48)
```

```python
# runs elsewhere: needs a translated DiffusionGemma checkpoint for the encoder
from dew.diffusion.block import denoise_logits, prefill_cache

cache = prefill_cache(encoder, variables, prompt)
logits = denoise_logits(decoder, sc, variables, sc_variables, cache, canvas, None)
```

## Multimodal wrappers

A gemma3, llama4, gemma4 or qwen3_5 checkpoint wraps three halves.
`translate_wrapper_config` turns it into decoder, tower and projector
records, and `translate_wrapper_weights` routes the released `model.*`
nesting into the three trees. The towers live in `dew.nn.vision`:

| Wrapper | Tower | Projector | Soft tokens per image |
| --- | --- | --- | --- |
| gemma3 | SigLIP trunk sharing CLIP's attention | Block average, norm, map to text width | Fixed by `mm_tokens_per_image` |
| llama4 | MetaCLIP-style trunk with grid rotary and pixel shuffle | Map of the shuffled output | Fixed by the grid and the shuffle ratio |
| gemma4 | Patch pixels with summed 2D tables, RMS blocks, position pooler | Scale-free norm and map | Varies with the image resolution |
| qwen3_5 | NaViT-style patchify with the frame repeated along time, resampled positions, merger reads the trunk | The merger itself | Varies with the image resolution |

Each projector is a registered value built by `projector_from_record`. A
gemma3n wrapper refuses: its tower is MobileNet-v5, an image classifier with
no counterpart here. The towers take one still image resolution per call;
packed batches of mixed resolutions and video frames have no form here.

```python
# runs elsewhere: downloads a Gemma 3 multimodal checkpoint from the Hub
from dew.interop.hf_decoders import translate_wrapper_config

record = translate_wrapper_config(wrapper)
```

A batch carries token ids with image marks beside the pixels. The soft tokens
land on the marks before the decoder reads them.

```python
import numpy as np

from dew.nn.vision import merge_soft_tokens

token_embeds = np.zeros((1, 6, 8), np.float32)
soft_tokens = np.ones((1, 2, 8), np.float32)
mask = np.array([[False, True, False, True, False, False]])
merged = merge_soft_tokens(token_embeds, soft_tokens, mask)
```

## Running it

```bash
python recipes/lm/train.py data:token-windows --data.path data/shakespeare-byte --data.seq-len 256 \
    --trainer.batch-size 32 --trainer.epochs 10 \
    --model.config '{"emb_features": 384, "num_layers": 6, "num_heads": 6}' \
    --sample-prompt "To be, or not to be" --sample-tokens 200
```

`--data.seq-len` is the context the model trains on; it reaches the model as its `max_seq_len`, which is also the size of the decode cache, so a `--sample-tokens` budget past the training context raises that limit to fit it. `--tokenizer` names the tokenizer `meta.json` was written with. `--balance-rate` turns on the aux-loss-free routing bias of a mixture-of-experts model, and `--mtp-weight` the multi-token-prediction term of a model with prediction depths. The rest is the shared configuration: `--optim.*` for the optimizer, `--trainer.mesh.fsdp` and `--trainer.accumulation` for scaling, `--trainer.wandb.project` to log.

Every logging tick writes `train/loss`, `train/ce` and `train/token_accuracy` (and `train/mtp_ce` with the term on) beside the trainer's throughput numbers. Every validation pass writes `val/perplexity` and, with a prompt configured, the generated text as a table.
