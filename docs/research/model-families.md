# Model families: what Dew's `CausalTransformer` needs for parity

Research note, 2026-09-02. Scope: the block-level inventory of the large open
model families, the smallest checkpoint of each that can act as a parity
fixture on a 16 GB card, and a dependency-ordered plan for Dew.

## How this was checked

Two primary sources only.

1. The model implementations in the pinned venv, transformers 5.16.1, at
   `~/Desktop/dew/.venv/lib/python3.12/site-packages/transformers`. Below this
   is written `TF`, so `TF/models/qwen3/configuration_qwen3.py:62` is a real
   file and line in this checkout. Files whose header says "generated from
   modular_x.py" are still the code that runs.
2. The `config.json`, `README.md` and safetensors headers of the actual
   checkpoints on the Hugging Face hub, fetched with
   `huggingface_hub.hf_hub_download(repo, 'config.json')` and
   `get_safetensors_metadata(repo)`. Every lineup below was listed with
   `HfApi().list_models(author=...)` before being written about. Parameter
   counts and byte sizes are the sums over the safetensors headers, not
   marketing numbers from a card.

Citation convention. `TF/...py:NNN` is a full path and line. A bare `(:NNN)`
means the same file as the last full path in that paragraph. Line numbers into
`TF` are stable because nothing edits site-packages. Dew's own code is cited by
symbol name, not by line, because `causal_transformer.py` is being edited on
other branches while this was written and any line number would be stale by the
time it is read.

Where the code was ambiguous the family's own card or report is cited. Where a
fact could not be established it says so, and the last section lists those.

Gating: every repo named in the fixture table was checked with
`HfApi().model_info(repo).gated`. Only Google's Gemma 3 repos returned
`manual`; `google/gemma-4-*`, `deepseek-ai/*`, `zai-org/*`, `moonshotai/*`,
`MiniMaxAI/*`, `openai/gpt-oss-*` and `Qwen/*` all returned `False`.
`meta-llama/Llama-4-Scout-17B-16E` returns 401 on an anonymous
`config.json` fetch, so it is gated too.

## Master table

Component by family. "Dew today" is the last row: what
`src/dew/nn/backbones/causal_transformer.py` already does.

| Family (model_type) | Token mixer | Norms | MLP | MoE router | MTP | Other |
|---|---|---|---|---|---|---|
| Qwen3 (`qwen3`) | GQA, full or sliding past `max_window_layers`, qk-norm on head dim | RMSNorm pre-attn and pre-MLP, plain scale | SwiGLU | none (dense); `qwen3_moe` softmax top-8 of 128 | no | rope theta 1e6, head_dim 128 fixed |
| Qwen3-Next (`qwen3_next`) | 3 GatedDeltaNet then 1 gated GQA, repeating | RMSNorm, gated RMSNorm inside the linear mixer | SwiGLU experts | softmax top-10 of 512, renormalised, plus sigmoid-gated shared expert | 1 layer, `mtp.*` | partial rotary 0.25, head_dim 256 |
| Qwen3.5 dense (`qwen3_5`) | same 3:1 GDN / gated attention, `full_attention_interval=4` | RMSNorm, qk-norm on head dim | SwiGLU dense | none | 1 layer, `mtp.*` | interleaved mRoPE `[11,11,10]`, vocab 248320, VL by default |
| Qwen3.5-MoE (`qwen3_5_moe`), also 3.6 and 3.8 MoE | as above | as above | SwiGLU experts | softmax top-k, always renormalised, sigmoid-gated shared expert | 1 layer | 256 or 512 experts |
| Qwen3.8-Flash-Next (`qwen4_exp`) | 3 GDN then 1 gated attention with a QSA block indexer | grouped RMSNorm over the 4 hyper-connection streams | SwiGLU experts | softmax top-10 of 512 | config field only | hyper-connections (`hc_count=4`), per-layer n-gram embeddings (PLE) |
| Gemma 3 (`gemma3_text`) | GQA, 5 sliding then 1 full, qk-norm | sandwich norms (4 per block), `(1+w)` scale | GeGLU (`gelu_pytorch_tanh`) | none | no | `query_pre_attn_scalar`, embedding scale, logit softcap, two rope thetas |
| Gemma 3n (`gemma3n_text`) | GQA with KV sharing over the last layers | sandwich, plus AltUp and LaurelBlock | GeGLU with activation sparsity | none | no | PLE per layer, `num_kv_shared_layers` |
| Gemma 4 (`gemma4_text`) | GQA, 5 sliding then 1 full, last layer forced full, `k_eq_v` option, KV sharing | sandwich norms, q/k/v norms, norm without scale on v | GeGLU, optional double-wide | softmax top-8 of 128 in parallel with the dense MLP | no | attention scale 1.0, `global_head_dim=512`, PLE, `layer_scalar` |
| DiffusionGemma (`diffusion_gemma`) | Gemma 4 MoE block, causal encoder plus bidirectional decoder | same as Gemma 4 | GeGLU + MoE | softmax top-8 of 128 | no | canvas 256, self-conditioning, softcap 30 |
| DeepSeek-V3 (`deepseek_v3`) | MLA: q lora 1536, kv lora 512, nope 128 + rope 64, v 128 | RMSNorm, plus norms on both loras | SwiGLU | sigmoid, `e_score_correction_bias`, group-limited 4 of 8, renormalised, `routed_scaling_factor` | 1 layer | interleaved rope, YaRN mscale into the attention scale, 3 dense layers first |
| DeepSeek-V3.2 (`deepseek_v32`) | MLA plus DSA lightning indexer, top-2048 keys | as V3, indexer key norm is a LayerNorm with bias | SwiGLU | as V3 | 1 layer | every layer sparse, FP8 weights with `weight_scale_inv` |
| DeepSeek-V4 (`deepseek_v4`) | CSA (compress 4) and HCA (compress 128) interleaved, MQA with 1 kv head, head_dim 512, sliding 128 branch, indexer top-512 | RMSNorm plus mHC Sinkhorn residual mixing | clipped SwiGLU (`swiglu_limit=10`) | `sqrtsoftplus` scores, top-6 of 256, plus 3 hash-MoE bootstrap layers | 1 layer in the checkpoint | grouped output projection (`o_groups`, `o_lora_rank`), two rope bases |
| GLM-4.7 (`glm4_moe`) | GQA, partial rotary 0.5 | RMSNorm, qk-norm | SwiGLU | sigmoid with `e_score_correction_bias`, 1 group, renormalised | 1 layer, `enorm`/`hnorm`/`eh_proj` | dense first layer only |
| GLM-4.7-Flash (`glm4_moe_lite`) | MLA (kv lora 512, q lora 768, v 256) | RMSNorm | SwiGLU | sigmoid + bias, top-4 of 64 | 1 layer | `routed_scaling_factor=1.8` |
| GLM-5.2 / 5.3 (`glm_moe_dsa`) | MLA (nope 192, rope 64, v 256) plus DSA indexer, `indexer_types` full or shared | RMSNorm | SwiGLU | sigmoid + bias, top-8 of 256, scaling 2.5 | 1 layer | indexer reuse across 4-layer groups |
| GLM-5.3-Flash (`glm5_next`) | 3 KDA linear then 1 MLA+DSA with k-pooling | RMSNorm, gated RMSNorm, unweighted-mean hyper head | clamped SwiGLU (10.0) | sigmoid + bias, top-8 of 288 | 1 layer | hyper-connections with Sinkhorn, `qk_rope_head_dim=0` (NoPE) |
| Kimi Linear (`kimi_linear`) | 3 KDA then 1 MLA with NoPE (`mla_use_nope`) | RMSNorm | SwiGLU | sigmoid, grouped top-8 of 256, scaling 2.446 | 0 | remote code on the hub, not in transformers 5.16.1 |
| Kimi K2.5 / K3 (`kimi_k25`, `kimi_k3`) | DeepSeek-V3 MLA text backbone | RMSNorm | SwiGLU | sigmoid, top-8 of 384 (K2.5) or 896 experts (K3) | 0 | `kimi_k25` maps its text config to `deepseek_v3` |
| MiniMax-Text-01 / M1 (`minimax`) | lightning attention alternating with full attention, per-block alpha/beta scaling | RMSNorm | SwiGLU | softmax top-2 of 8 | no | decay slopes per head |
| MiniMax-M2 (`minimax_m2`) | full GQA every layer, qk-norm over the whole projection | RMSNorm | SwiGLU experts | sigmoid top-8 of 256, no renormalisation | no | FP8 weights |
| MiniMax-M3 (`minimax_m3_vl`) | GQA, partial rotary 0.5 | RMSNorm | clamped SwiGLU (7.0) | sigmoid + `e_score_correction_bias`, top-4 of 128, shared expert | 1 layer | 1M context |
| gpt-oss (`gpt_oss`) | GQA with attention sinks, alternating sliding 128 and full | RMSNorm | `swigluoai`: clamped, alpha 1.702, `(up+1)` term, interleaved packing, expert biases | top-4 of 32 or 128, softmax **after** top-k, router bias | no | MXFP4 blocks and scales, qkvo biases |
| Llama 4 (`llama4_text`) | GQA, chunked attention 8192, NoPE every 4th layer with temperature tuning, L2 qk-norm | RMSNorm | SwiGLU, wider on dense layers | softmax top-1 of 16 plus a shared expert | no | `attn_temperature_tuning` |
| Mixtral (`mixtral`) | GQA | RMSNorm | SwiGLU | softmax then top-2 of 8, renormalised | no | the reference simple MoE |
| Nemotron-H (`nemotron_h`) | Mamba2 blocks with a few attention and MLP blocks, pattern string | RMSNorm | ReLU squared, no gate | optional sigmoid top-2 of 8 | field exists, 0 in the small checkpoints | `hybrid_override_pattern` |
| **Dew today** | GQA, full or sliding, qk-norm, rotate-half rope | RMSNorm pre-attn and pre-MLP, optional `(1+w)` | SwiGLU or GeGLU | none | no | embedding scale, logit softcap, two rope thetas |

## Per-family notes

### Qwen3, the dense baseline

`Qwen3Config` is 20 fields (`TF/models/qwen3/configuration_qwen3.py:62-84`).
`head_dim` is a config field, 128, not `hidden_size // heads`. Sliding
attention is off unless `use_sliding_window` is set, and `__post_init__` nulls
`sliding_window` when it is off (`:87`). When it is on, layers from
`max_window_layers` up are sliding, which is the opposite direction from
Gemma. `Qwen/Qwen3-0.6B/config.json` has 28 layers, 16 query heads, 8 kv
heads, `head_dim=128`, `rope_theta=1000000`, `tie_word_embeddings=true`,
`use_sliding_window=false`.

This family is already inside Dew's field set. Nothing new is needed.

[skip] Already covered by `CausalTransformer`. The Qwen3-0.6B parity fixture
belongs to the HfDecoders branch, not here.

### Qwen3-Next: GatedDeltaNet, and the shape of every later Qwen

`Qwen3NextConfig` adds one block of linear-attention fields
(`TF/models/qwen3_next/configuration_qwen3_next.py:109-113`):
`linear_conv_kernel_dim=4`, `linear_key_head_dim=128`,
`linear_value_head_dim=128`, `linear_num_key_heads=16`,
`linear_num_value_heads=32`. `partial_rotary_factor` defaults to 0.25
(`:129`), so with `head_dim=256` only 64 dims rotate.

`Qwen3NextGatedDeltaNet` (`TF/models/qwen3_next/modeling_qwen3_next.py:512-699`):

- Two input projections. `in_proj_qkvz` produces q, k, v and the output gate z
  in one matrix, `in_proj_ba` produces the two scalars per value head (`:540-543`).
- A depthwise causal conv1d over the concatenated q, k, v, `groups=conv_dim`,
  kernel 4, no bias (`:530-537`).
- Two learned per-head vectors: `dt_bias` and `A_log` (`:547-551`).
- `beta = sigmoid(b)` and `g = -exp(A_log) * softplus(a + dt_bias)`, computed
  in fp32 to stop `-inf` in fp16 (`:651-653`).
- The recurrence is the gated delta rule, chunked for prefill
  (`torch_chunk_gated_delta_rule`, `:375`) and a single-step recurrence for
  decode (`:457`), with q and k L2-normalised inside the kernel
  (`use_qk_l2norm_in_kernel=True`, `:668`).
- The output goes through a gated RMSNorm, `norm(x) * silu(z)` in fp32
  (`Qwen3NextRMSNormGated`, `:58-74`), then `out_proj`.

The MoE block is a softmax top-k router with renormalisation
(`Qwen3NextTopKRouter`, `:758-776`) and a shared expert whose output is scaled
by `sigmoid(shared_expert_gate(x))` (`:779-798`). Experts are stored as two 3D
tensors, `gate_up_proj [E, 2*I, H]` and `down_proj [E, H, I]` (`:727-728`).

`Qwen/Qwen3-Next-80B-A3B-Instruct/config.json`: 48 layers, `hidden_size=2048`,
512 experts, 10 active, `moe_intermediate_size=512`,
`shared_expert_intermediate_size=512`, `head_dim=256`,
`linear_num_value_heads=32`. Its safetensors carry MTP weights under `mtp.*`
(`mtp.fc.weight [2048, 4096]`, `mtp.pre_fc_norm_embedding`,
`mtp.pre_fc_norm_hidden`, one full decoder layer, `mtp.norm`).

[borrow: reimplement the idea] GatedDeltaNet is the single highest-value new
mixer. It plugs into the `mixer` slot of `DecoderBlock`
(`DecoderBlock.mixer` in `src/dew/nn/backbones/causal_transformer.py`) with no change to the
block. The chunked delta rule is about 80 lines of JAX with
`jax.lax.scan` over chunks; the decode path is a second, shorter function.
The conv1d state and the recurrent state both have to live in the same cache
collection Dew already uses for keys and values.

### Qwen3.5, 3.6, 3.8: one architecture, four sizes, mRoPE by default

`HfApi().list_models(author='Qwen')` on 2026-09-02 returns, without
quantised or GGUF copies:

| Repo | model_type | Layers | Hidden | Experts |
|---|---|---|---|---|
| `Qwen/Qwen3.5-0.8B`, `-2B`, `-4B`, `-9B`, `-27B` (and `-Base` for the first four) | `qwen3_5` | 24 to 64 | 1024 to 5120 | dense |
| `Qwen/Qwen3.5-35B-A3B`, `-122B-A10B`, `-397B-A17B` | `qwen3_5_moe` | 40+ | 2048+ | 256 |
| `Qwen/Qwen3.6-27B`, `Qwen/Qwen3.6-35B-A3B` | `qwen3_5` / `qwen3_5_moe` | 64 / 40 | 5120 / 2048 | dense / 256 |
| `Qwen/Qwen3.8-27B`, `Qwen/Qwen3.8-2.4T-A95B` | `qwen3_5` / `qwen3_5_moe_text` | 64 / 92 | 5120 / 8192 | dense / 512 |
| `Qwen/Qwen3.8-Flash-Next` | `qwen4_exp` | 48 | 2560 | 512 |

So Qwen 3.6 and 3.8 reuse the 3.5 classes. There is no `Qwen4-*` repo under
the `Qwen` author, and no `qwen4_exp` checkpoint other than
`Qwen3.8-Flash-Next`.

Three things are new against Qwen3-Next.

**Gated attention.** `Qwen3_5Attention` projects
`num_heads * head_dim * 2` from `q_proj`, splits it into the query and a gate,
and multiplies the attention output by `sigmoid(gate)` before `o_proj`
(`TF/models/qwen3_5/modeling_qwen3_5.py:645-704`). `q_norm` and `k_norm` are
RMSNorms over `head_dim` only (`:656-657`). The config flag is
`attn_output_gate: true` in every 3.5, 3.6 and 3.8 `config.json`.

**Interleaved mRoPE.** `mrope_section = [11, 11, 10]` and the frequency layout
is interleaved `THWTHWTHW...TT` rather than sectioned
(`TF/models/qwen3_5/modeling_qwen3_5.py:101,149-166`). Text-only positions
collapse to plain rope, so a text parity test does not need the vision path,
but the frequency order does have to match.

**Dense MLP on every layer for `qwen3_5`.** `Qwen3_5DecoderLayer` builds
`Qwen3_5MLP(config, config.intermediate_size)` unconditionally (`:752`); only
`qwen3_5_moe` swaps in the sparse block. The `qwen3_5_moe` router drops the
`norm_topk_prob` flag and always renormalises
(`TF/models/qwen3_5_moe/modeling_qwen3_5_moe.py:776`).

`Qwen/Qwen3.5-0.8B/config.json`: 24 layers, `hidden_size=1024`,
`intermediate_size=3584`, 8 query heads and 2 kv heads at `head_dim=256`,
`linear_num_key_heads=16`, `linear_num_value_heads=16`,
`layer_types` = three `linear_attention` then one `full_attention`, six times,
`vocab_size=248320`, `tie_word_embeddings=true`,
`rope_theta=10000000`, `partial_rotary_factor=0.25`,
`max_position_embeddings=262144`. The card states the same layout, "6 x (3 x
(Gated DeltaNet -> FFN) -> 1 x (Gated Attention -> FFN))", and "MTP: trained
with multi-steps" (`Qwen/Qwen3.5-0.8B/README.md`, blog
`https://qwen.ai/blog?id=qwen3.5`, license apache-2.0).

The safetensors of `Qwen/Qwen3.5-0.8B` confirm the MTP head is one decoder
layer plus three norms and a fusion matrix: `mtp.fc.weight [1024, 2048]`,
`mtp.pre_fc_norm_embedding`, `mtp.pre_fc_norm_hidden`,
`mtp.layers.0.*` (a full gated-attention block), `mtp.norm`. The linear-attention
tensors are `in_proj_qkv [6144, 1024]`, `in_proj_z [2048, 1024]`,
`in_proj_a [16, 1024]`, `in_proj_b [16, 1024]`, `conv1d.weight [6144, 1, 4]`,
`dt_bias [16]` (bf16), `A_log [16]` (fp32), `norm.weight [128]` (fp32),
`out_proj [1024, 2048]`. Note that 3.5 splits q, k, v, z and the two scalars
into four projections, `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`
(`TF/models/qwen3_5/modeling_qwen3_5.py:427-430`), where Qwen3-Next packed
them into two, `in_proj_qkvz` and `in_proj_ba`
(`TF/models/qwen3_next/modeling_qwen3_next.py:542-543`). A loader written
against one will not open the other.

[adopt: depend on it] for the numbers: use `transformers` as the reference in
the parity test, not a hand-copied config.
[borrow: reimplement the idea] for the code: gated attention is a two-line
change inside a `CausalSelfAttention` variant (double the q projection, gate
the output). Interleaved mRoPE is a change to `rotary_freqs`
(`rotary_freqs` in `src/dew/nn/backbones/causal_transformer.py`) that takes 3D positions.
Seam: a new mixer factory next to `CausalSelfAttention`, selected by
`layer_types`.

### Qwen3.8-Flash-Next (`qwen4_exp`): hyper-connections, PLE, a query-sparse indexer

This is the most different model in the whole survey. Three mechanisms that no
other family in transformers 5.16.1 has all at once.

**Hyper-connections.** The residual stream is `hc_count=4` streams wide, so the
hidden state carries `4 * hidden_size` features.
`Qwen4ExpTextGatedResidual` (`TF/models/qwen4_exp/modeling_qwen4_exp.py:941-969`)
normalises the wide state with a grouped RMSNorm, computes a low-rank mixing
weight through `hc_lowrank=320` with `silu` then `sigmoid`, averages the four
streams into one input for the block, and returns per-stream injection weights
`2 * sigmoid(...)`. The block output is broadcast back over the streams
(`:1236-1243`). Every layer has two of these, one before attention and one
before the MLP.

**Per-layer embeddings from hashed n-grams.** `Qwen4ExpTextPLELayer`
(`:1117-1191`) and `Qwen4ExpTextNGramEmbedding` (`:1018-1116`) hash token
n-grams (`ngram_size=3`, `heads_per_ngram=8`) into per-layer tables sized from
`ngram_vocab_size_base=20_000_000`, with multipliers derived from a splitmix64
seed (`:979-1017`), then a dilated depthwise conv. PLE is only allowed on
`linear_attention` layers and only on the layer ids in `ple_layer_ids`
(validation at `TF/models/qwen4_exp/configuration_qwen4_exp.py:248-257`).
`Qwen/Qwen3.8-Flash-Next/config.json` sets `ple_layer_ids: [2]`. This one table
is why the repo is 180B parameters and 360 GB on disk while the compute path is
small; the config comment says the embedding is around 45B parameters and is
sharded on dim 0 across 512 shards (`configuration_qwen4_exp.py:97-99,157`).

**QSA indexer.** `Qwen4ExpTextQSAIndexer` (`:611-719`) projects one query set
and a single key head (`indexer_kv_heads` must be 1), averages
`indexer_compress_ratio=4` consecutive keys into blocks, scores blocks, keeps
`indexer_budget // compress_ratio` of them plus the incomplete tail, and turns
that into a mask that is ANDed into the attention mask (`:793-797`). The config
enforces `rotary_dim <= indexer_head_dim`
(`configuration_qwen4_exp.py:225-231`). Checkpoint values:
`indexer_n_heads=4`, `indexer_kv_heads=1`, `indexer_head_dim=128`,
`indexer_budget=2048`, `indexer_compress_ratio=4`.

Note that `layer_types` in the checkpoint says `full_attention`, and
`__post_init__` rewrites those entries to `qwen_sparse_attention`
(`configuration_qwen4_exp.py:180-184`). A config translator in Dew has to do the
same rewrite or it will build the wrong layer.

The card links a tech report PDF at
`https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf` and a
blog at `https://qwen.ai/blog?id=qwen3.8-flash-next`; the repo license field is
`other`.

[later] All three mechanisms. Hyper-connections change the residual contract of
every block, so they cannot hide behind the `mixer` slot. PLE needs a second
input path carrying `input_ids` down the stack. The indexer needs a masked
attention kernel. None of this is reachable as a parity test on 16 GB anyway:
the smallest checkpoint is 360 GB. Revisit if a small `qwen4_exp` appears.

### Gemma 3

`Gemma3TextConfig` (`TF/models/gemma3/configuration_gemma3.py:77-102`):
`query_pre_attn_scalar=256`, `sliding_window=4096`,
`final_logit_softcapping`, `attn_logit_softcapping`,
`use_bidirectional_attention`, and
`default_theta = {"global": 1e6, "local": 1e4}` (`:102`).

Four facts a port must match exactly.

1. The attention scale is `query_pre_attn_scalar ** -0.5`, not
   `head_dim ** -0.5` (`TF/models/gemma3/modeling_gemma3.py:318`). They happen
   to be equal at 256, but the field is what the checkpoint carries.
2. `q_norm` and `k_norm` are RMSNorms over `head_dim`, applied before rope
   (`:338-339,356-357`).
3. Sandwich norms: `input_layernorm`, `post_attention_layernorm`,
   `pre_feedforward_layernorm`, `post_feedforward_layernorm`, four per block
   (`:396-397,424-426`). Dew has two.
4. The embedding is scaled by a buffer, not recomputed:
   `Gemma3TextScaledWordEmbedding` multiplies by `embed_scale` cast to the
   weight dtype (`:111-117`). Dew casts `sqrt(d)` to the activation dtype
   (the `embedding_scale` branch of `CausalTransformer.__call__`). At bf16 those differ.

`google/gemma-3-*` repos are gated (`gated='manual'`), so an anonymous
`config.json` fetch returns 401. This is a real obstacle for CI.

[borrow: reimplement the idea] Sandwich norms and `query_pre_attn_scalar` are
two fields plus two norm calls in `DecoderBlock`. The HfDecoders branch owns
the Gemma 3 translation; this note only records that the gating means the parity
fixture needs a token or a locally cached copy.

### Gemma 3n

Kept for the record, since two of its mechanisms reappear in Gemma 4.

- `num_kv_shared_layers`: the last N layers do not own `k_proj`/`v_proj` and
  read the keys and values of the last non-sharing layer of the same type
  (`TF/models/gemma3n/modeling_gemma3n.py:1178-1186,1233-1234`).
- Per-layer embeddings: `per_layer_input_gate` and `per_layer_projection` per
  block (`:1287-1293`).
- `AltUp` (`:989`) and `LaurelBlock` (`:933`), plus activation sparsity on the
  MLP (`:961-973`).

[skip] AltUp and Laurel are Gemma 3n only. KV sharing and PLE are worth doing,
and they are worth doing in the Gemma 4 shape, below. Seam, when they are done
there: KV sharing needs a shared cache slot that `DecoderBlock` can read across
layers, and PLE needs `CausalTransformer.__call__` to pass the token ids down
the stack. Neither fits the `mixer` factory as it stands.

### Gemma 4 and the E-series

Lineup, all ungated: `google/gemma-4-E2B`, `-E4B`, `-12B`, `-31B`,
`-26B-A4B`, each with an `-it` variant, plus `-26B-A4B-it-assistant`,
`-12B-it-assistant` and QAT copies. `gemma-4-12B` reports
`model_type=gemma4_unified`; E2B, E4B, 31B and 26B-A4B report `gemma4`.

`Gemma4TextConfig` (`TF/models/gemma4/configuration_gemma4.py:152-183`) plus
`__post_init__` (`:185-225`):

- Layer pattern is 5 sliding then 1 full (`sliding_window_pattern = 6`), and
  the last layer is forced to `full_attention` with a warning (`:190-201`).
- Rope is per layer type: sliding uses `rope_type=default`, theta 1e4; full
  uses `rope_type=proportional`, `partial_rotary_factor=0.25`, theta 1e6
  (`:203-208`). Partial rotary on the global layers is new against Gemma 3.
- Full-attention layers get overrides through `per_layer_config`:
  `head_dim = global_head_dim = 512` and, when `attention_k_eq_v` is set,
  `num_key_value_heads = num_global_key_value_heads` (`:210-223`). So the
  global layers have a different head_dim from the sliding layers in the same
  model. `gemma-4-26B-A4B` sets `num_global_key_value_heads=2`.
- `sliding_window` is halved and one is added when
  `use_bidirectional_attention == "all"`, and `to_dict` undoes it (`:186-188,227-234`).

`Gemma4TextAttention` (`TF/models/gemma4/modeling_gemma4.py:1166-1279`):

- `self.scaling = 1.0` (`:1181`). The scale is folded into the trained weights.
  A port that uses `head_dim ** -0.5` will be wrong by a factor of 16 at
  head_dim 256.
- `q_norm`, `k_norm`, and a `v_norm` **without a scale parameter** (`:1196-1201`).
  Norming the values is not something any other family here does.
- `attention_k_eq_v` on non-sliding layers drops `v_proj` and reuses the key
  projection output as the value (`:1206-1212,1247`).
- KV sharing: layers at or past `num_hidden_layers - num_kv_shared_layers`
  own no k/v weights and read `shared_kv_states[layer_type]`, which is written
  by the last non-sharing layer of that type (`:1186-1191,1240-1259`).
  `gemma-4-E2B` shares 20 of 35 layers, `E4B` shares 18 of 42.

`Gemma4RMSNorm` (`:197-215`) normalises in fp32, multiplies by
`weight.float()`, then casts back. There is no `(1+w)` offset, and the weights
ship centred on 1. This is a change from Gemma 3 and it means Dew's
`scale_offset` flag must be **off** for Gemma 4 and on for Gemma 3.

`Gemma4TextDecoderLayer` (`:1359-1445`) is the most decorated block in the
survey: four norms, an optional MoE branch that runs **in parallel** with the
dense MLP and is summed (`:1418-1430`), a per-layer input gate and projection
with its own norm (`:1435-1442`), and a `layer_scalar` buffer multiplying the
block output (`:1444`).

The router (`Gemma4TextRouter`, `:1322-1356`) is unlike every other router
here: a norm without scale, then a learned `scale` vector times
`hidden_size ** -0.5`, then a linear projection, then softmax in fp32, top-k,
renormalise, then multiply by a learned `per_expert_scale` gathered at the
selected indices.

Embedding scales are buffers: `hidden_size ** 0.5` for the token embedding and
`hidden_size_per_layer_input ** 0.5` for the PLE table (`:1589,1604-1608`).
`final_logit_softcapping=30.0` in every checkpoint config.

[borrow: reimplement the idea] Gemma 4 needs, in order: sandwich norms,
`scaling=1.0` and per-layer-type head_dim, v-norm, KV sharing, PLE, then the
parallel MoE branch. The first three are field work in `DecoderBlock` and
`CausalSelfAttention`. KV sharing needs the block to be able to read another
block's cache, which is the first thing in this document that Dew's current
`mixer`-only seam cannot express.

### DiffusionGemma

`google/diffusiongemma-26B-A4B-it`, `model_type=diffusion_gemma`,
`architectures: ['DiffusionGemmaForBlockDiffusion']`, apache-2.0, 25.8B
parameters, 51.6 GB bf16, ungated. Its text config is the Gemma 4 26B-A4B
config with `canvas_length=256` added at the top level and
`final_logit_softcapping=30.0` fixed as a class attribute
(`TF/models/diffusion_gemma/configuration_diffusion_gemma.py:99`).

It is an encoder-decoder over the same weights shape:

- `DiffusionGemmaEncoderTextAttention` takes `is_causal` from
  `use_bidirectional_attention != "all"`
  (`TF/models/diffusion_gemma/modeling_diffusion_gemma.py:281`). The encoder
  runs over the prompt and fills a KV cache.
- `DiffusionGemmaDecoderTextAttention` hard-codes `is_causal = False` (`:383`).
  The decoder attends over the canvas and the cached prompt, both directions.
- The decoder mask is built by `create_diffusion_decoder_attention_mask`
  (`:1326-1440`), with query length equal to `canvas_length` and key length
  equal to cache plus canvas, using `bidirectional_mask_function` and
  `allow_is_causal_skip=False`.
- `DiffusionGemmaSelfConditioning` (`:790-823`) turns the previous step's
  logits into soft embeddings through a gated MLP with a pre-norm and a
  scale-free post-norm, and adds them to the input embeddings.
- The head divides by 30, applies tanh, multiplies by 30, in fp32 (`:1666-1670`).

There is no timestep embedding and no mask token. The sampler starts from a
canvas of **uniform random token ids**
(`TF/models/diffusion_gemma/generation_diffusion_gemma.py:394-404`), accepts
tokens by an entropy bound, and anneals a temperature linearly with the step
index (`:276-316`). `generation_config.json` on the hub:
`max_denoising_steps=48`, `t_max=0.8`, `t_min=0.4`,
`confidence_threshold=0.005`, `entropy_bound=0.1`, `canvas_length` from the
model config. The card confirms the design: "autoregressive encoder to process
and cache the prompt context, paired with a decoder that applies bidirectional
attention over the generation canvas", "8 active experts out of 128"
(`google/diffusiongemma-26B-A4B-it/README.md`, blog at
`https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/`).

[later] The mask plumbing is the interesting part for Dew and it is cheap: a
flag that makes the attention mask bidirectional over a suffix of the sequence.
The rest of DiffusionGemma is a sampler, which belongs in `src/dew/sampling`,
and a Gemma 4 backbone, which is the Gemma 4 work above.

### DeepSeek V3, V3.2, V4

`deepseek-ai` on the hub has, in order of recency: `DeepSeek-V4-Flash-Vision-Exp`,
`DeepSeek-V4-Flash-0731`, `DeepSeek-V4-Pro-0813`, `DeepSeek-V4-Flash`,
`DeepSeek-V4-Pro`, `DeepSeek-V4-Flash-DSpark`, `DeepSeek-V4-Flash-Base`,
`DeepSeek-V4-Pro-Base`, `DeepSeek-V3.2`, plus the older V3, V2 and V2-Lite. All
ungated, MIT license on the V4 cards.

**MLA.** `DeepseekV3Attention`
(`TF/models/deepseek_v3/modeling_deepseek_v3.py:361-494`):

| Piece | Shape source | Note |
|---|---|---|
| `q_a_proj` then `q_a_layernorm` then `q_b_proj` | `hidden -> q_lora_rank -> heads * (nope + rope)` | skipped when `q_lora_rank is None`, then a single `q_proj` |
| `kv_a_proj_with_mqa` | `hidden -> kv_lora_rank + qk_rope_head_dim` | one matrix for the latent and the shared rope key |
| `kv_a_layernorm` | `kv_lora_rank` | applied to the latent only, not to `k_rot` |
| `kv_b_proj` | `kv_lora_rank -> heads * (nope + v_head_dim)` | expands to per-head keys and values |
| `o_proj` | `heads * v_head_dim -> hidden` | note `v_head_dim != qk_head_dim` |

Order matters in three places. The rope slice is rotated before the cache write
(`:463-471`), and the cache stores the **compressed** latent plus the single
rope key, not the expanded keys and values. `rope_interleave=True` selects
`apply_rotary_pos_emb_interleave` instead of the rotate-half convention
(`:464-467`). The attention scale is `qk_head_dim ** -0.5` multiplied by the
YaRN mscale squared when rope scaling is on
(`yarn_apply_mscale`, `:417` and `:280-289`).

**Router.** `DeepseekV3TopkRouter` (`:131-170`): logits in fp32, `sigmoid`,
add `e_score_correction_bias` (a buffer, shape `[n_routed_experts]`, fp32 in
the checkpoints), score each of `n_group` groups by the sum of its top 2,
keep `topk_group` groups, mask the rest to `-inf`, take top-k, gather the
**unbiased** sigmoid scores for the selected experts, renormalise if
`norm_topk_prob`, multiply by `routed_scaling_factor`. Every detail of that
sequence is load-bearing and easy to get subtly wrong.

`deepseek-ai/DeepSeek-V3.2/config.json`: 61 layers, hidden 7168,
128 heads, `q_lora_rank=1536`, `kv_lora_rank=512`, `qk_rope_head_dim=64`,
`qk_nope_head_dim=128`, `v_head_dim=128`, 256 experts, 8 active, 1 shared,
`n_group=8`, `topk_group=4`, `routed_scaling_factor=2.5`,
`first_k_dense_replace=3`, `scoring_func=sigmoid`,
YaRN with `factor=40`, `original_max_position_embeddings=4096`,
`mscale=1.0`, `mscale_all_dim=1.0`, `num_nextn_predict_layers=1`,
`index_topk=2048`, `index_head_dim=128`, `index_n_heads=64`.

**DSA indexer.** `DeepseekV32Indexer`
(`TF/models/deepseek_v32/modeling_deepseek_v32.py:160-256`) is small and
self-contained: `wq_b` from the query lora rank to `index_n_heads *
index_head_dim`, `wk` from hidden to one `index_head_dim` key,
`k_norm` a **LayerNorm with bias** (not RMSNorm), `weights_proj` from hidden to
`index_n_heads`. Scores are `relu(q . k * head_dim ** -0.5)` in fp32, weighted
per head by `weights_proj(x) * n_heads ** -0.5`, summed over heads, masked, then
top-`index_topk`. The transformers docstring records two deliberate
simplifications against the reference: no Hadamard rotation (orthogonal, so dot
products are preserved) and no FP8 scoring kernel (`:191-207`). The indexer uses
the half-split rope convention while the main MLA uses interleaved (`:232`).

**V4.** `DeepseekV4Config`
(`TF/models/deepseek_v4/configuration_deepseek_v4.py:40-97` for the field
documentation, `:139-191` for the values) describes a different model again:

- `layer_types` from `{compressed_sparse_attention, heavily_compressed_attention}`
  with `compress_rates` 4 and 128 (`:46-55,159`). Each block also has a sliding
  window branch with `sliding_window=128` (`:76-78`).
- MQA: `num_key_value_heads=1`, `head_dim=512`, `partial_rotary_factor`
  defaults to `64/512` (`:144-147`).
- Grouped output projection with `o_groups` and `o_lora_rank` (`:79-83`).
- Manifold-constrained hyper-connections, `hc_mult=4`, with
  `hc_sinkhorn_iters=20` Sinkhorn-Knopp iterations projecting the residual
  mapping onto doubly stochastic matrices (`:59-66`).
- `scoring_func='sqrtsoftplus'`, top-6 of 256, and `mlp_layer_types` from
  `{hash_moe, moe}` where the first 3 layers route by a frozen
  `tid2eid[input_ids]` lookup (`:42-43,67-73`).
- `swiglu_limit=10.0` clipping the routed experts' pre-activations (`:74-75`).
- Two rope bases, `rope_theta=10000` for the sliding branch and
  `compress_rope_theta=160000` with YaRN for the compressed branches, and the
  YaRN `attention_factor` forced to 1.0 because the reference does not apply
  mscale (`:294-321`).

`deepseek-ai/DeepSeek-V4-Flash/config.json` matches: 43 layers, hidden 4096,
64 heads, 1 kv head, head_dim 512, `q_lora_rank=1024`, `index_topk=512`,
`o_groups=8`, and a legacy `compress_ratios` list
`[0, 0, 4, 128, 4, 128, ... 4, 0]` which `__post_init__` maps to `layer_types`
(`:266-276`). V4-Pro is 61 layers, hidden 7168, 384 experts, `o_groups=16`,
`index_topk=1024`. The card cites `https://arxiv.org/abs/2606.19348`.

**MTP.** transformers does not put MTP layers in the model classes. It builds
them on demand: `MtpLayer` (`TF/modeling_layers.py:316-361`) is
`enorm(embeds)` and `hnorm(previous_hidden)` concatenated, then
`eh_proj: 2*hidden -> hidden`, then one copy of the family's decoder layer,
then an optional post-norm. `MtpModel` (`:364-426`) ties the embedding and the
`lm_head` to the main model and is used by `MTPCandidateGenerator`
(`TF/generation/candidate_generator.py:1423`) for speculative decoding.
The V3.2 safetensors carry exactly that, as layer 61:
`model.layers.61.embed_tokens.weight [129280, 7168]`, `enorm`, `hnorm`,
`eh_proj [7168, 14336]`, `shared_head.norm`, `shared_head.head [129280, 7168]`.
Qwen instead names them `mtp.fc` and `mtp.pre_fc_norm_{embedding,hidden}`, and
GLM-4.7-Flash uses the DeepSeek names at layer 47.

**FP8.** V3.2 ships `F8_E4M3` weights with a paired
`*.weight_scale_inv` fp32 tensor per matrix, blocked 128x128
(a `[18432, 7168]` weight has a `[144, 56]` scale). Dequantisation is
`w.astype(f32) * scale[i // 128, j // 128]`. V4-Flash is more mixed: 141.7B
`I8`, 8.9B `F8_E8M0`, 6.0B `F8_E4M3`, 1.4B `BF16`.

[later] for MLA and DSA as trainable mixers, [borrow] for the router.
The sigmoid-plus-bias group-limited router is the router that Qwen, GLM, Kimi,
MiniMax and DeepSeek all converge on, so it is worth writing once and well.
MLA is a self-contained mixer and fits the `mixer` slot; the only awkward part
is that the decode cache holds latents of a different shape from Dew's current
`[B, S, K, D]` key cache. DSA needs a top-k mask fed into
`scaled_dot_product_attention`, which
`scaled_dot_product_attention` in `src/dew/nn/attention.py` already accepts
through its `mask` argument.

### GLM

`zai-org` lineup on 2026-09-02: `GLM-5.3`, `GLM-5.3-Flash`, `GLM-5.3-BF16`,
`GLM-5.3-Flash-BF16`, `GLM-5.2`, `GLM-5.1`, `GLM-5`, `GLM-4.7`,
`GLM-4.7-Flash`, `GLM-4.6V-Flash`, plus the 4.5 and 4-0414 generations. All
ungated, MIT on the 5.3 cards.

Four distinct architectures behind those names:

| Repo | model_type | Attention | Router |
|---|---|---|---|
| `GLM-4.7` | `glm4_moe` | GQA, 96 heads, 8 kv, partial rotary 0.5 | sigmoid + bias, 160 experts, top-8 |
| `GLM-4.7-Flash` | `glm4_moe_lite` | MLA, `q_lora=768`, `kv_lora=512`, v 256, nope 192, rope 64 | sigmoid + bias, 64 experts, top-4, scaling 1.8 |
| `GLM-5.2`, `GLM-5.3` | `glm_moe_dsa` | MLA + DSA, 78 layers, `q_lora=2048`, nope 192, rope 64, v 256, `index_n_heads=32`, `index_topk=2048` | sigmoid + bias, 256 experts, top-8, scaling 2.5 |
| `GLM-5.3-Flash` | `glm5_next` | 3 KDA linear then 1 MLA+DSA, 45 layers, `qk_rope_head_dim=0` | sigmoid + bias, 288 experts, top-8 |

`GLM-5.3` adds an `indexer_types` list that alternates `full` and `shared`:
`['full','full','full','shared','shared','shared','full','shared',...]`, so one
indexer's top-k selection is reused by the next three MLA layers
(`TF/models/glm5_next/configuration_glm5_next.py:44-46`). That is a cheap idea
worth noting: the indexer is the expensive part of DSA and it is shared 4:1.

`GLM-5.3-Flash` is the interesting one for Dew because it combines almost
everything:

- **KDA linear attention.** `Glm5NextTextLinearAttention`
  (`TF/models/glm5_next/modeling_glm5_next.py:584-733`), documented in the
  source as "Kimi-style KDA (Kimi Linear Attention)". Separate `q_proj`,
  `k_proj`, `v_proj` at `linear_head_dim=128` times `linear_num_heads=64`, one
  depthwise conv over the concatenation, a low-rank forget gate, a per-head
  `beta = sigmoid(b_proj(x))`, a low-rank output gate `g_b_proj(g_a_proj(x))`,
  a gated RMSNorm whose activation is **sigmoid** not silu (`:344`), then
  `o_proj`.
- **Forget gate.** `Glm5NextTextForgetGate` (`:305-335`): low-rank
  `f_b_proj(f_a_proj(x))` plus `dt_bias`, times `exp(A_log)`, and when
  `linear_lower_bound` is set the gate is
  `lower_bound * sigmoid(decay_rate * g)` instead of the softplus form. The
  checkpoint leaves `linear_lower_bound` unset, so the softplus branch with the
  `g > 20` guard runs.
- **Hyper-connections with Sinkhorn.** `Glm5NextTextHyperConnection`
  (`:219-296`) does `hc_sinkhorn_iters=20` row and column normalisations, and
  `Glm5NextTextHyperHead` (`:298-302`) collapses the streams by an unweighted
  mean, which the source explicitly contrasts with DeepSeek-V4. The checkpoint
  carries `hc_attn_base [24]`, `hc_attn_scale [3]`, `hc_attn_fn [24, 16384]`
  and the ffn equivalents per layer.
- **Clamped SwiGLU.** `gate.clamp(max=swiglu_limit)` and
  `up.clamp(-limit, limit)` with `swiglu_limit=10.0` (`:96-103,139-140`).
- **k-pooled indexer.** `Glm5NextTextIndexer` (`:736-1024`) pools
  `index_kpool` consecutive keys with a learned positional code
  (`index_kpool_compress_ape [4, 128]`) and a gate
  (`index_kpool_compress_gate [128, 4096]`), selects
  `index_topk // index_kpool` pools, and always includes the incomplete tail
  when `index_kpool_always_select_tail`. The checkpoint sets `index_kpool=4`
  while the class default is 16.

The card cites `https://arxiv.org/abs/2602.15763` and
`https://z.ai/blog/glm-5.3-flash`.

[borrow: reimplement the idea] KDA is close enough to GatedDeltaNet that one
JAX delta-rule kernel with a gate-shape parameter covers both. Do GatedDeltaNet
first, then KDA is a small delta: separate q/k/v projections, low-rank gates,
sigmoid gated norm. Seam: a second mixer factory beside the GatedDeltaNet one,
selected by `layer_types`, plus the clamp option on `GatedMLP` for
`swiglu_limit`.
[later] the hyper-connections and the k-pooled indexer. Hyper-connections have
no seam in Dew today: they change the residual contract that `DecoderBlock`
owns, so they would need a wide-residual block variant rather than a new
mixer.

### Kimi

`moonshotai` lineup: `Kimi-K3`, `Kimi-K2.7-Code`, `Kimi-K2.6`, `Kimi-K2.5`,
`Kimi-Linear-48B-A3B-Instruct` and `-Base`, `Kimi-K2-Thinking`,
`Kimi-K2-Base`, `Kimi-K2-Instruct`, plus VL and audio models. All ungated, MIT.

`Kimi-K2.5` reports `model_type=kimi_k25`. Its wrapper config maps the text
config's model type to `deepseek_v3`, including the legacy `kimi_k2` name
(`TF/models/kimi_k25/configuration_kimi_k25.py:83-92`). So the Kimi K2 text
backbone **is** DeepSeek-V3 MLA plus a sigmoid router:
`Kimi-K2.5/config.json` has `kv_lora_rank=512`, `q_lora_rank=1536`,
`qk_rope_head_dim=64`, `qk_nope_head_dim=128`, `v_head_dim=128`, 384 experts,
8 active, 1 shared, `routed_scaling_factor=2.827`,
`first_k_dense_replace=1`, `rope_theta=50000`, 61 layers, vocab 163840.
`Kimi-K3` is the same family scaled up: 93 layers, 896 experts,
`max_position_embeddings=1048576`.

`Kimi-Linear-48B-A3B-*` has `model_type=kimi_linear` with an `auto_map`
pointing at `configuration_kimi.KimiLinearConfig` and
`modeling_kimi.KimiLinearModel`, so it runs from remote code, not from
transformers 5.16.1 (there is no `models/kimi_linear` directory in this
venv). Its config gives the layout directly:
`linear_attn_config = {'full_attn_layers': [4, 8, 12, 16, 20, 24, 27],
'head_dim': 128, 'kda_layers': [1,2,3,5,6,7,...,26],
'num_heads': 32, 'short_conv_kernel_size': 4}`, with `mla_use_nope=True`,
`kv_lora_rank=512`, `q_lora_rank=None`, `v_head_dim=128`,
`moe_router_activation_func='sigmoid'`, `moe_renormalize=True`,
`num_experts=256`, `num_experts_per_token=8`,
`routed_scaling_factor=2.446`, `num_nextn_predict_layers=0`. The card describes
KDA as "a refined version of Gated DeltaNet that introduces a more efficient
gating mechanism", cites `https://huggingface.co/papers/2510.26692`, and points
at the reference kernels in
`https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda`.

Note `mla_use_nope=True` with `qk_rope_head_dim=64` still in the config: the
3:1 KDA to MLA layout puts all position information in the linear layers, so
the full-attention layers use no rope at all. GLM-5.3-Flash does the same by
setting `qk_rope_head_dim=0`.

[borrow: reimplement the idea] KDA, shared with GLM-5.3-Flash. Same seam: a
mixer factory selected by `layer_types`, with the NoPE case handled by letting
that factory skip rope entirely rather than by setting `rope_theta` to
something large.
[skip] the K2/K3 text backbone as a separate item; it is DeepSeek-V3 MLA and
falls out of the MLA work, so its seam is the MLA mixer in step 7.

### MiniMax

`MiniMaxAI` lineup: `MiniMax-H3`, `MiniMax-M3`, `MiniMax-M2.7`, `M2.5`,
`M2.1`, `M2`, `MiniMax-Text-01`, `MiniMax-VL-01`, `M1-40k`, `M1-80k`, plus
`MiniMax-M3-MXFP8`. `MiniMax-H3` has no `config.json` at the repo root (404),
so nothing is claimed about it here.

`minimax` (Text-01, M1) is the only lightning-attention model in transformers.
`MiniMaxLightningAttention`
(`TF/models/minimax/modeling_minimax.py:122-265`) with `get_slope_rate` and
`decay_factors`, `block_size=256`, and six scalar mixing factors in the config
(`full_attn_alpha_factor`, `full_attn_beta_factor`,
`linear_attn_alpha_factor`, `linear_attn_beta_factor`, `mlp_alpha_factor`,
`mlp_beta_factor`, `TF/models/minimax/configuration_minimax.py:112-118`),
alternating full and linear layers every other layer (`:126`).

`MiniMax-M2` went back to plain full attention on every layer:
`MiniMaxM2Attention` (`TF/models/minimax_m2/modeling_minimax_m2.py:287-348`)
is GQA with `q_norm` and `k_norm` over the **whole** projection
(`num_heads * head_dim` and `num_kv_heads * head_dim`, `:303-304`), not per
head. The checkpoint tensors agree: `q_norm.weight [6144]`,
`k_norm.weight [1024]` with `head_dim=128`, 48 heads, 8 kv heads. Its router
(`:46-64`) is sigmoid with **no** renormalisation, and the checkpoint carries
`block_sparse_moe.e_score_correction_bias [256]` and an fp32
`block_sparse_moe.gate.weight [256, 3072]`. Expert tensors are named
`w1`/`w2`/`w3`, not `gate_proj`/`up_proj`/`down_proj`.

`MiniMax-M3` reports `model_type=minimax_m3_vl`,
`architectures: ['MiniMaxM3SparseForConditionalGeneration']`: hidden 6144,
60 layers, 64 heads, 4 kv heads, `partial_rotary_factor=0.5`, 128 experts,
top-4, 1 shared expert, `routed_scaling_factor=2.0`, `swiglu_limit=7.0`,
`num_nextn_predict_layers=1`, `max_position_embeddings=1048576`.

[skip] lightning attention. It is one family, one generation old, and that
lab's own next model dropped it. Record it and move on.
[borrow] the M2 whole-projection qk-norm as a flag on the qk-norm, since it is
a one-line difference and it silently changes the numbers. Seam: the `qk_norm`
option inside `CausalSelfAttention`, which today norms per head. Also the
clamp for M3's `swiglu_limit=7.0`, which is the same `GatedMLP` option GLM
needs.

### gpt-oss

`openai/gpt-oss-20b` and `-120b`, plus two safeguard variants. Ungated.

Two things Dew does not have.

**Attention sinks.** Each attention module owns a learned
`sinks` parameter of shape `[num_attention_heads]`
(`TF/models/gpt_oss/modeling_gpt_oss.py:293`). In the forward, the sink logit
is concatenated to the attention logits as one extra key, the max is subtracted
for stability, softmax runs over the widened axis, and the sink column is
dropped before multiplying by the values (`:251-259`). The effect is a learned
escape valve: attention probabilities no longer have to sum to one over real
tokens. The comment records that the max subtraction is not in the original
implementation and slightly changes results.

**A router that softmaxes after top-k.** `GptOssTopKRouter` (`:117-130`)
takes `topk` over the raw logits and then softmaxes over the k selected values.
Every other family softmaxes or sigmoids first. It also has a router **bias**
(`router.bias [32]` in the 20b checkpoint).

Layer pattern alternates sliding 128 and full
(`layer_types` in `config.json`, and `sliding_window=128` at
`TF/models/gpt_oss/configuration_gpt_oss.py:52`). `attention_bias=True`
(`:68`), so q, k, v and o all have biases, which the checkpoint confirms.
Experts have biases too: `gate_up_proj_bias [32, 5760]`,
`down_proj_bias [32, 2880]`.

**A third thing: the expert activation is not plain SwiGLU.** `_apply_gate`
(`:82-88`) splits the packed projection by **interleaving**,
`gate_up[..., ::2]` and `gate_up[..., 1::2]`, clamps `gate` above at 7.0 and
`up` to `[-7, 7]`, then computes `(up + 1) * gate * sigmoid(gate * 1.702)`.
The alpha and the limit are hard-coded in `__init__` (`:79-80`), not config
fields. MiniMax M3 uses the same formula from config fields and its checkpoint
calls it `swigluoai`. The `(up + 1)` term and the interleaved packing are both
easy to miss and neither one fails loudly.

**MXFP4.** The expert weights ship as
`gate_up_proj_blocks [32, 5760, 90, 16]` and
`gate_up_proj_scales [32, 5760, 90]`, both `U8`. Each byte of the blocks tensor
holds two fp4 values, so 90 * 16 * 2 = 2880 input features per row, and the
scale is one E8M0 exponent per group of 32 values. Dequantising 20b to bf16
gives about 21B parameters, so 42 GB. That is the reason gpt-oss cannot be a
16 GB fixture without splitting the model.

[borrow: reimplement the idea] Attention sinks are five lines inside
`scaled_dot_product_attention` in `src/dew/nn/attention.py`: widen the
logits by one column per head before the softmax and drop it after. They only
work on the reference path, not on a fused kernel, so the flag has to force the
reference implementation. Worth doing: sinks are cheap, they are in a
frontier-lab open model, and they change training stability claims.
[later] the MXFP4 loader.

### Llama 4

`Llama4TextConfig` (`TF/models/llama4/configuration_llama4.py:140-174`) has
four ideas worth recording:

- `no_rope_layers` with `no_rope_layer_interval=4`: every fourth layer uses no
  positional encoding at all (`TF/models/llama4/modeling_llama4.py:326,344-348`).
- `attn_temperature_tuning` on those NoPE layers: queries are scaled by
  `log1p(floor((pos + 1) / floor_scale)) * attn_scale + 1` with
  `floor_scale=8192` and `attn_scale=0.1` (`:366-374`).
- `attention_chunk_size=8192` with a chunked causal mask
  (`create_chunked_causal_mask`, `:29,550`). Chunked is not sliding: tokens
  attend inside their own block only.
- L2 qk-norm (`Llama4TextL2Norm`), applied only when the layer uses rope, and
  present only on the 16E model (`:362-364`).

The MoE is top-1 of 16 with a shared expert, `interleave_moe_layer_step`
deciding which layers are sparse (`:359-371`), and dense layers use the wider
`intermediate_size_mlp`.

`meta-llama/Llama-4-Scout-17B-16E` is gated: an anonymous `config.json` fetch
returns 401.

[skip] for now. Chunked attention and NoPE layers are each a small flag, but no
other family here needs them, the smallest checkpoint is 109B, and it is gated.
Revisit only if a Llama 4 parity fixture becomes a requirement. If it does, the
seams are small: chunked masking goes in `causal_attention_mask` next to
`sliding_window`, the NoPE and temperature-tuning switches go in
`CausalSelfAttention`, and the L2 qk-norm is a third option on the qk-norm
flag.

### Mixtral

The reference simple MoE, and the right first target for Dew's MoE work:
`MixtralTopKRouter` (`TF/models/mixtral/modeling_mixtral.py:96-111`) is
softmax over all experts, then top-k, then renormalise. No bias, no groups, no
scaling factor, no shared expert. `num_local_experts=8`,
`num_experts_per_tok=2` (`TF/models/mixtral/configuration_mixtral.py:86-87`).

`mistralai/Mixtral-8x7B-v0.1` is 46.7B parameters, 93.4 GB bf16, ungated. There
is no small Mixtral, so the parity fixture for a plain MoE has to be a random
init compared against `MixtralSparseMoeBlock` on CPU.

[borrow: reimplement the idea] Write the MoE FFN against this router first,
then add the sigmoid, bias, group-limit and scaling options that DeepSeek, GLM,
Kimi and MiniMax need. Seam: `DecoderBlock.mlp`, which is currently a hardwired
`GatedMLP` (`DecoderBlock.setup` in `src/dew/nn/backbones/causal_transformer.py`). It needs the
same factory treatment `mixer` already has.

### Nemotron-H

`NemotronHConfig` (`TF/models/nemotron_h/configuration_nemotron_h.py:87-133`)
is a Mamba2 hybrid: `layers_block_type` (the checkpoints ship
`hybrid_override_pattern`, a string like
`M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-...` where `M` is Mamba2, `*` is attention and
`-` is MLP), `ssm_state_size=128`, `mamba_num_heads`, `mamba_head_dim`,
`n_groups=8`, `conv_kernel=4`, `chunk_size`, and time-step clamps. The MLP
activation is `relu2`, squared ReLU with no gate (`:104`), which no other
family here uses. `NemotronHMamba2Mixer`
(`TF/models/nemotron_h/modeling_nemotron_h.py:356-540`) delegates to the
`mamba_ssm` kernels with a torch fallback.

`nvidia/Nemotron-H-4B-Base-8K`: 4.49B parameters, 9.0 GB bf16, ungated, 52
layers, hidden 3072, `mamba_num_heads=112`, `mamba_head_dim=64`.
`nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` is 3.97B with the same `nemotron_h`
type and a denser attention pattern.

Dew already has an SSM implementation at `src/dew/nn/ssm.py`, used by
`ssm_dit.py`.

[later] A Mamba2 mixer is a real piece of work and the delta-rule family is
where the open decoders went. If the SSM in `src/dew/nn/ssm.py` turns out to be
close to Mamba2 already, revisit; that check is not part of this note.

## Vocabulary, context, activation, tying

Every value here is read from the checkpoint's own `config.json`.

| Checkpoint | Vocab | Native context | MLP activation | Tied embeddings |
|---|---|---|---|---|
| `Qwen/Qwen3-0.6B` | 151936 | 40960 | silu | yes |
| `Qwen/Qwen3.5-0.8B` | 248320 | 262144 | silu | yes |
| `google/gemma-4-E2B` | 262144 | 131072 | `gelu_pytorch_tanh` | yes |
| `deepseek-ai/DeepSeek-V3.2` | 129280 | 163840 | silu | no |
| `deepseek-ai/DeepSeek-V4-Flash` | 129280 | 1048576 | silu | no |
| `zai-org/GLM-4.7` | 151552 | 202752 | silu | no |
| `zai-org/GLM-5.3-Flash` | 154880 | 1048576 | silu | no |
| `moonshotai/Kimi-K2.5` | 163840 | 262144 | silu | no |
| `moonshotai/Kimi-Linear-48B-A3B-Instruct` | 163840 | 1048576 | silu | no |
| `MiniMaxAI/MiniMax-M2` | 200064 | 196608 | silu | no |
| `MiniMaxAI/MiniMax-M3` | 200064 | 1048576 | `swigluoai` | no |
| `openai/gpt-oss-20b` | 201088 | 131072 | silu | no |
| `nvidia/Nemotron-H-4B-Base-8K` | 131072 | 8192 | `relu2` | no |
| `GSAI-ML/LLaDA-8B-Base` | 126464 | 4096 | silu | field absent |
| `Dream-org/Dream-v0-Base-7B` | 152064 | 131072 | silu | no |

Four observations that matter for Dew.

**Only the small dense models tie their embeddings.** Every MoE model in this
table unties, so `tie_embeddings=False` and a real `lm_head` is the common case
for anything large. Dew defaults to tied
(`CausalTransformer.tie_embeddings`), which is right for from-scratch small
training and wrong for every checkpoint it might load.

**Vocabularies are large and padded.** Qwen3.5's 248320 is described on the
card as "248320 (Padded)". At `hidden_size=1024` the embedding is 254M
parameters out of 873M, so 29 percent of that checkpoint is the embedding.
An fp32 logits tensor at batch 1, sequence 128 is 127 MB, which is fine, but
at sequence 2048 it is 2.0 GB, so a parity test should use short sequences.

**`swigluoai` and `relu2` are activations Dew does not have, and there are two
different clipped SwiGLUs, not one.** Dew's `GatedMLP` supports `swiglu` and
`geglu` only. The variants, all read from the code:

| Name | Formula | Values | Used by |
|---|---|---|---|
| `swiglu` | `down(silu(gate) * up)` | none | most families |
| clamped `swiglu` | `down(silu(clamp(gate, max=L)) * clamp(up, -L, L))` | L = 10.0 | GLM-5.3-Flash, GLM-5.x (`TF/models/glm5_next/modeling_glm5_next.py:99-106`) |
| `swigluoai` | `down((clamp(up, -L, L) + 1) * g * sigmoid(g * a))` where `g = clamp(gate, max=L)` | a = 1.702, L = 7.0 | gpt-oss (`TF/models/gpt_oss/modeling_gpt_oss.py:79-88`), MiniMax M3 (`TF/models/minimax_m3_vl/modeling_minimax_m3_vl.py:173-179`) |
| `relu2` | `down(relu(x) ** 2)`, no gate at all | none | Nemotron-H |

Three traps here. The `(up + 1)` term in `swigluoai` is easy to miss and it
shifts the whole function, so GLM's clamped SwiGLU and gpt-oss's are not the
same activation despite both being described as clipped. `a = 1.702` is the
sigmoid approximation of GELU, so `swigluoai` is closer to a clamped GeGLU than
to a SwiGLU. And in gpt-oss the packed `gate_up_proj` **interleaves** gate and
up, `gate_up[..., ::2]` and `gate_up[..., 1::2]`
(`TF/models/gpt_oss/modeling_gpt_oss.py:83`), where everyone else who packs
the two projections splits it in halves with `chunk(2, dim=-1)`, MiniMax M3
included (`TF/models/minimax_m3_vl/modeling_minimax_m3_vl.py:175`). A loader
that assumes halves for gpt-oss will silently interleave the two projections,
and the model will still run.

MiniMax M3's `config.json` declares `hidden_act: "swigluoai"`, and
transformers rewrites it to `silu` in `__post_init__` because the real gate is
computed inline from `swiglu_alpha` and `swiglu_limit` and `hidden_act` has to
be a valid `ACT2FN` key
(`TF/models/minimax_m3_vl/configuration_minimax_m3_vl.py:126-128`). So a
translator that reads `hidden_act` from the config file and a translator that
reads it from a constructed config object will disagree.

**Dew's `geglu` is already the right gelu for Gemma.** Gemma 3 and Gemma 4 use
`gelu_pytorch_tanh`. Dew's `GatedMLP` calls flax's `nn.gelu(gate)`, and
`jax.nn.gelu` defaults to `approximate=True`, which is the tanh form. Measured
on this machine at fp32 on `[-1.0, 0.3, 2.0]`, the jax default and PyTorch's
`0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715 x^3)))` agree to 2.98e-08 max absolute
difference. So this is one Gemma detail that needs no work and no caveat.

**Tokenizers.** Qwen 3.5 ships its own class,
`Qwen3_5Tokenizer(TokenizersBackend)` with `model = BPE`, NFC normalisation,
a `Split` on `PRETOKENIZE_REGEX` then byte level with
`add_prefix_space=False`, and `<|endoftext|>` as unk, eos and pad
(`TF/models/qwen3_5/tokenization_qwen3_5.py:28-92`). No other family in this
survey needs a new tokenizer class in transformers 5.16.1. Dew tokenises
through `tools/tokenize_text.py` and `src/dew/data/text.py`, so the tokenizer
is a data-side concern and not part of the block inventory. It does matter for
one thing: a parity test that starts from text rather than from token ids will
differ if the pre-tokenizer regex differs, so parity tests should feed token
ids directly.

[borrow: reimplement the idea] Two small changes fall out of this table. Add a
`clamp` limit to `GatedMLP`, which GLM-5.x, MiniMax M3 and DeepSeek V4 all
need, and add an ungated squared-ReLU option if Nemotron-H is ever attempted.
Seam: `GatedMLP` in `src/dew/nn/backbones/causal_transformer.py`.
[skip] the tokenizers. They are a data-side concern and Dew already has
`src/dew/data/text.py`; parity tests should take token ids, not text.

## Fixture table

Sizes are the sum over the safetensors headers, measured on 2026-09-02.
"Fits 16 GB bf16" means the weights plus a batch-1, 128-token forward with fp32
logits fit a 16 GB card with room for the runtime. The card in this workstation
is a 16 GB RTX 4080, so the rule of thumb below is: weights under about 11 GB
are comfortable, 11 to 14 GB needs care, over 14 GB needs the CPU backend.

**This column is about a forward pass, not about training.** A parity test runs
one forward with no optimizer state, so the budget is roughly weights plus
activations. Training the same checkpoint needs several times that. Measured on
this workstation by the HfDecoders branch on 2026-09-02: continued pretraining
of `Qwen/Qwen3-0.6B`, whose weights are 1.5 GB in bf16, peaks at 11.82 GB for
batch 1 at sequence 512 with AdamW, fp32 parameters and EMA, and does not fit
at batch 4 (XLA asks for 14.67 GiB, cannot rematerialise below 11.54 GiB, and
dies on a 5.19 GiB allocation). That is about eight times the weight size for
the smallest model in this table. So read a "yes" below as "a parity forward
fits", never as "this is trainable here".

One correction for whoever writes these tests: the allocator knob is
`XLA_CLIENT_MEM_FRACTION`. `XLA_PYTHON_CLIENT_MEM_FRACTION` still works but is
deprecated in this jaxlib, and setting both raises
(`~/Desktop/dew/.venv/lib/python3.12/site-packages/jaxlib/xla_client.py:180-190`).
Preallocation is turned off with `XLA_PYTHON_CLIENT_PREALLOCATE` (`:191`). The
default fraction is set inside the C++ allocator, not in the Python layer, so
its value was not verified here; measure it with
`jax.local_devices()[0].memory_stats()['bytes_limit']` on an idle GPU rather
than trusting a remembered number.

| Family | Smallest checkpoint | Params | Disk | Ship dtype | Gated | Fits 16 GB bf16 | Fits fp32 |
|---|---|---|---|---|---|---|---|
| Qwen3 dense | `Qwen/Qwen3-0.6B` | 0.75B | 1.5 GB | bf16 | no | yes | yes |
| Qwen3.5 dense (GDN + gated attn) | `Qwen/Qwen3.5-0.8B-Base` | 0.87B | 1.7 GB | bf16 | no | yes | yes |
| Qwen3.5 dense, next size | `Qwen/Qwen3.5-2B` | 2.27B | 4.5 GB | bf16 | no | yes | yes (9.1 GB) |
| Qwen3.5 dense, largest that fits | `Qwen/Qwen3.5-4B` | 4.66B | 9.3 GB | bf16 | no | yes, tight | no |
| Qwen3.5 MoE | `Qwen/Qwen3.5-35B-A3B` | 35.95B | 71.9 GB | bf16 | no | no | no |
| Qwen3-Next | `Qwen/Qwen3-Next-80B-A3B-Instruct` | 81.3B | 162.6 GB | bf16 | no | no | no |
| Qwen4-Exp | `Qwen/Qwen3.8-Flash-Next` | 180B (mostly the PLE table) | 360 GB | bf16 | no | no | no |
| Gemma 3 | `google/gemma-3-1b-pt` | not measured, repo gated | n/a | bf16 | yes, manual | expected yes | expected yes |
| Gemma 4 E-series | `google/gemma-4-E2B` | 5.12B | 10.2 GB | bf16 | no | yes, tight | no |
| Gemma 4 E-series, next | `google/gemma-4-E4B` | 8.00B | 16.0 GB | bf16 | no | no | no |
| Gemma 4 dense | `google/gemma-4-12B` | 11.96B | 23.9 GB | bf16 | no | no | no |
| Gemma 4 MoE | `google/gemma-4-26B-A4B` | 25.81B | 51.6 GB | bf16 | no | no | no |
| DiffusionGemma | `google/diffusiongemma-26B-A4B-it` | 25.82B | 51.6 GB | bf16 | no | no | no |
| DeepSeek MLA, small | `deepseek-ai/DeepSeek-V2-Lite` | 15.71B | 31.4 GB | bf16 | no | no | no |
| DeepSeek V3.2 (DSA) | `deepseek-ai/DeepSeek-V3.2` | 685B | 689 GB | FP8 E4M3 + scale_inv | no | no | no |
| DeepSeek V4 | `deepseek-ai/DeepSeek-V4-Flash` | 158B | 160 GB | I8 + E8M0 + E4M3 + bf16 | no | no | no |
| GLM MoE | `zai-org/GLM-4.7-Flash` | 31.2B | 62.4 GB | bf16 | no | no | no |
| GLM KDA + DSA | `zai-org/GLM-5.3-Flash` | 321B | 328 GB | FP8 E4M3 | no | no | no |
| Kimi Linear (KDA) | `moonshotai/Kimi-Linear-48B-A3B-Base` | 49.1B | 98.2 GB | bf16 | no | no | no |
| MiniMax | `MiniMaxAI/MiniMax-M2` | 229B | 230 GB | FP8 E4M3 | no | no | no |
| gpt-oss | `openai/gpt-oss-20b` | 12.0B counted (about 21B logical) | 13.8 GB | MXFP4 blocks + scales | no | no, 42 GB dequantised | no |
| Llama 4 | `meta-llama/Llama-4-Scout-17B-16E` | not measured, repo gated | n/a | bf16 | yes | no | no |
| Mixtral | `mistralai/Mixtral-8x7B-v0.1` | 46.7B | 93.4 GB | bf16 | no | no | no |
| Nemotron-H | `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` | 3.97B | 7.9 GB | bf16 | no | yes | no |
| Nemotron-H, older | `nvidia/Nemotron-H-4B-Base-8K` | 4.49B | 9.0 GB | bf16 | no | yes, tight | no |

Three consequences.

**Only four families have a real single-card fixture.** Qwen3, Qwen3.5 dense,
Gemma 4 E2B, and Nemotron-H. Everything else is a CPU test or a partial test.

**There is no torch in this venv, and that decides the shape of every parity
test.** `transformers` is a declared runtime dependency of Dew
(`pyproject.toml`), but `torch` is not, and it is not installed:
`import torch` raises `ModuleNotFoundError`. On import, transformers prints
"PyTorch was not found. Models won't be available and only tokenizers,
configuration and file/data utilities can be used." So the reference **configs**
are available today and the reference **models** are not. Three options, in
order of preference:

1. Commit reference tensors. A generator script under `tools/` runs the
   reference block once in an environment that does have torch, writes small
   inputs and outputs to `tests/fixtures/`, and the everyday test compares Dew
   against those files with no torch and no network. A separate, network and
   torch marked test regenerates them. This is the only option that keeps the
   default test run fast and dependency free.
2. Add torch as a test extra next to `test = ["pytest", "safetensors"]`. This
   makes the comparison live and self checking, at the cost of a large
   dependency in CI and two frameworks in one process.
3. Compare against the equations rather than the code. Acceptable only for
   pieces whose reference is a short formula, and not for anything with a
   projection layout to get wrong.

Either way the **config translation** can be tested today with no new
dependency: build the real `Qwen3Config` or `Gemma4TextConfig`, let its
`__post_init__` derive `layer_types`, `rope_parameters` and `per_layer_config`,
then assert Dew's translated fields match. That catches the whole class of bugs
where a checkpoint says `full_attention` and the family rewrites it, as
`qwen4_exp` does, or where the sliding pattern runs in the opposite direction,
as Qwen3 and Gemma do.

This was run in Dew's own venv on 2026-09-02 and it works with no torch. The
output below is real, not illustrative:

```python
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

Qwen3Config(num_hidden_layers=6, use_sliding_window=True,
            sliding_window=128, max_window_layers=4).layer_types
# ['full_attention'] * 4 + ['sliding_attention'] * 2

Gemma4TextConfig(num_hidden_layers=12).layer_types
# 5 sliding, then full, 5 sliding, then full: index 5 and index 11
Gemma4TextConfig(num_hidden_layers=12).rope_parameters
# {'sliding_attention': {'rope_type': 'default', 'rope_theta': 10000.0},
#  'full_attention': {'rope_type': 'proportional',
#                     'partial_rotary_factor': 0.25, 'rope_theta': 1000000.0}}
Gemma4TextConfig(num_hidden_layers=12).per_layer_config[5].head_dim
# 512, the global-layer override, against head_dim 256 on the sliding layers

Qwen4ExpTextConfig(num_hidden_layers=8,
                   layer_types=['linear_attention'] * 3 + ['full_attention']
                             + ['linear_attention'] * 3 + ['full_attention']
                   ).layer_types
# the two 'full_attention' entries come back as 'qwen_sparse_attention'
```

Note the two directions in the first two cases. Qwen3 makes the layers **from**
`max_window_layers` sliding, so the sliding layers are at the end. Gemma 4 makes
every sixth layer full, so the full layers are the exceptions. A translator that
gets this backwards still produces a valid model and quietly wrong numbers.

**For the numeric half, test one block, not one model.** Instantiate the
family's block class (`DeepseekV3Attention`, `Qwen3NextGatedDeltaNet`,
`GptOssAttention`, `Glm5NextTextLinearAttention`, `DeepseekV3TopkRouter`) with a
small config at fp32 on CPU, copy its `state_dict` into the Dew module, feed
the same input, compare. No checkpoint, no gate, no GPU, and it pins exactly
the thing being ported. A second test then loads the real checkpoint's first
two layers and compares hidden states, which is where a naming or transpose
error shows up.

**FP8 and MXFP4 dequantisation is a loader concern, not a model concern.** For
DeepSeek V3.2 and GLM-5.3-Flash it is
`w.astype(f32) * scale_inv[i // 128, j // 128]`. For gpt-oss it is
unpack two fp4 nibbles per byte, then scale by the E8M0 exponent per 32
values. Both belong next to `src/dew/interop/safetensors_io.py`, which today
does no renaming and no casting by design (see its module docstring).

[borrow: reimplement the idea] Build the fixture generator first, before any
of the block work below. Every step in the plan names a parity test, and none
of those tests can run until there is a way to get reference numbers into the
repo. Seam: a new script under `tools/` writing to `tests/fixtures/`, read by
the tests next to `tests/test_causal_transformer.py`.

## Implementation plan for Dew, ordered by dependency

Dew's seams today, for reference:

- `mixer` is a factory that `DecoderBlock` calls with a name and then calls as
  `mixer(x, decode=...)` (`DecoderBlock` in `src/dew/nn/backbones/causal_transformer.py`).
  Any mixer that needs positions, a mask, or another layer's cache does not fit
  this signature.
- `mlp` is a hardwired `GatedMLP`, built in `DecoderBlock.setup`. Not a
  factory yet.
- `layer_types` is a tuple validated in `CausalTransformer.setup` against the
  module constant `LAYER_TYPES`, which currently holds only `full_attention`
  and `sliding_attention`.
- The attention kernel is one function with a `mask` argument
  (`scaled_dot_product_attention` in `src/dew/nn/attention.py`), which is
  where sinks and top-k masks go.
- The decode cache is opened per module in the flax style
  (`open_kv_cache` in `src/dew/nn/attention.py`).

### Step 0: widen the two seams. Small.

Make `mlp` a factory like `mixer`, and let `LAYER_TYPES` be extended by the
caller rather than being a module constant. Give the mixer signature the two
arguments it will need for everything below: absolute positions, and an
optional mask. Nothing new is built here; this is the change that stops every
later step from touching `DecoderBlock` again.

Parity test: the existing `tests/test_causal_transformer.py` suite still
passes unchanged, in particular
`test_param_tree_mirrors_the_hf_decoder_layout` and
`test_decode_cache_matches_the_full_sequence`.

### Step 1: MoE FFN with a router family. Medium.

One `MoeMLP` module holding stacked expert weights,
`gate_up_proj [E, 2*I, H]` and `down_proj [E, H, I]`, matching the layout
transformers now uses everywhere (`Qwen3NextExperts`, `DeepseekV3Experts`,
`Gemma4TextExperts`), plus an optional shared expert with an optional sigmoid
gate, plus one router with these options:

| Option | Needed by |
|---|---|
| softmax over all, then top-k, renormalise | Mixtral, Qwen3-MoE, Qwen3-Next, Qwen3.5-MoE |
| renormalisation as an option, not an assumption | `Qwen3MoeConfig.norm_topk_prob` defaults to `False` (`TF/models/qwen3_moe/configuration_qwen3_moe.py:106`) while `Qwen/Qwen3-30B-A3B` and `Qwen/Qwen3-235B-A22B` both set it `true`, and `qwen3_5_moe` dropped the flag and always renormalises. Read it from the checkpoint, never from the class default. |
| top-k first, then softmax over the k, router bias | gpt-oss |
| sigmoid, `e_score_correction_bias`, renormalise, `routed_scaling_factor` | DeepSeek V3/V3.2, GLM all, Kimi, MiniMax M2/M3 |
| group-limited: score groups by their top-2 sum, keep `topk_group` | DeepSeek V3/V3.2 |
| learned input scale and `per_expert_scale`, MoE parallel to the dense MLP | Gemma 4 |
| `sqrtsoftplus` scores | DeepSeek V4 |
| clipped SwiGLU inside the expert | GLM-5.x, MiniMax M3, DeepSeek V4 |
| dense first `first_k_dense_replace` layers | DeepSeek, GLM, Kimi |

Unlocks: Mixtral, Qwen3-MoE, Qwen3.5-MoE, gpt-oss (with step 3), Gemma 4 MoE,
and the FFN half of every DeepSeek, GLM, Kimi and MiniMax model.

Parity test, no download: build `MixtralSparseMoeBlock` and
`DeepseekV3MoE` from `transformers` at fp32 on CPU with 8 experts and hidden
64, copy the state dict, assert identical argmax over expert indices and a
stated max absolute output difference. The router is the part that must match
bit for bit in its **selection**; a tie in the top-k is the one place where a
tiny numeric difference changes the output a lot, so the test should use a seed
whose scores are well separated and say so.

### Step 2: GatedDeltaNet mixer, and KDA behind the same kernel. Large.

The delta-rule recurrence, chunked for training and single-step for decode,
with q and k L2-normalised, a depthwise causal conv over q, k, v, a per-head
`beta`, a log-space decay `g`, and a gated output norm. Two gate shapes:
Qwen's `A_log` and `dt_bias` per value head with silu on the output gate, and
GLM/Kimi's low-rank `f_a_proj`/`f_b_proj` forget gate with sigmoid on the
output gate.

This is the step that needs a new cache kind: a conv state of shape
`[B, conv_dim, kernel - 1]` and a recurrent state of shape
`[B, heads, k_dim, v_dim]`, both written every step. Dew's
`open_kv_cache` in `src/dew/nn/attention.py` is the place that has to
learn about non-KV state.

Unlocks: Qwen3-Next, Qwen3.5, Qwen3.6, Qwen3.8 dense (with step 4),
Kimi-Linear, GLM-5.3-Flash's linear layers.

Parity test: `Qwen3NextGatedDeltaNet` from `transformers` at fp32 on CPU,
random init, copied state dict, one forward at sequence 64 and one 16-step
decode against the prefill. Then the network-marked test against
`Qwen/Qwen3.5-0.8B-Base` layer 0, which is a `linear_attention` layer, at fp32.
That checkpoint is 1.7 GB and ungated, so this is the one non-trivial mixer in
the whole survey that has a real single-card fixture.

### Step 3: attention sinks and the qk-norm variants. Small.

Sinks: one parameter per head, concatenated as an extra logit column before the
softmax, dropped after (`TF/models/gpt_oss/modeling_gpt_oss.py:251-258`). Only
on the reference path. Also here: qk-norm over the whole projection instead of
per head (MiniMax M2), and the L2 qk-norm variant (Llama 4), both one-line
options on the existing qk-norm.

Unlocks: gpt-oss's attention, MiniMax M2's attention.

Parity test: `GptOssAttention` at fp32 on CPU with 4 heads, comparing against
the Dew mixer with the same weights, and a second test asserting that the sink
column changes the output, so the test can fail if the sink is dropped.

### Step 4: gated attention output and interleaved mRoPE. Small.

`q_proj` outputs `2 * heads * head_dim`, split into query and gate, output
multiplied by `sigmoid(gate)`
(`TF/models/qwen3_5/modeling_qwen3_5.py:645-704`). Plus partial rotary, which
Dew does not have: only the first `partial_rotary_factor * head_dim` dims
rotate. Plus interleaved mRoPE for the 3D case, which for text-only positions
must reduce exactly to plain rope.

Unlocks, together with step 2: `Qwen/Qwen3.5-0.8B`, `-2B`, `-4B`, `-9B`,
`-27B`, `Qwen3.6-27B`, `Qwen3.8-27B`. That is the whole modern Qwen dense line.

Parity test: full-model logits against `Qwen3_5ForConditionalGeneration`'s text
tower at fp32 on `Qwen/Qwen3.5-0.8B-Base`, identical argmax and a stated max
absolute logit difference. This is the first end-to-end frontier-model parity
Dew can actually run on this workstation.

### Step 5: Gemma 4's block. Medium.

Sandwich norms (four per block), `scaling = 1.0`, per-layer-type head_dim
through a per-layer config, a scale-free v-norm, `attention_k_eq_v`, the
`layer_scalar` buffer, and RMSNorm **without** the `(1+w)` offset. Then KV
sharing, which needs a block to read another block's keys and values, which is
the second cache change: a shared slot keyed by layer type, written by the last
non-sharing layer (`TF/models/gemma4/modeling_gemma4.py:1186-1191,1240-1259`).

Unlocks: `google/gemma-4-E2B` and `-E4B` (with PLE, step 6), and the dense
`gemma-4-12B`/`-31B` shape without PLE.

Parity test: `google/gemma-4-E2B` at bf16 on the GPU or fp32 on CPU, first two
layers' hidden states, then full logits. Also a unit test that a KV-shared
layer produces the same keys as its source layer, so the sharing itself can
fail the test rather than being invisible.

### Step 6: per-layer embeddings. Medium.

A second embedding table of shape
`[vocab_size_per_layer_input, num_layers * hidden_size_per_layer_input]`,
scaled by `sqrt(hidden_size_per_layer_input)`, sliced per layer, then per
block: `per_layer_input_gate`, the activation, multiply by the per-layer input,
`per_layer_projection`, `post_per_layer_input_norm`, residual add
(`TF/models/gemma4/modeling_gemma4.py:1435-1442,1604-1608`). This is the step
that requires the token ids to reach every block, not just the embedding, which
is a change to `CausalTransformer.__call__`'s contract.

Unlocks: Gemma 4 E2B and E4B exactly (`hidden_size_per_layer_input=256`),
Gemma 3n if it is ever wanted. Note `gemma-4-26B-A4B` sets
`hidden_size_per_layer_input=0`, so the MoE model does not use PLE.

Parity test: full logits on `google/gemma-4-E2B`.

### Step 7: MLA mixer. Large.

Two loras on the query path with a norm between, one fused
`kv_a_proj_with_mqa` producing the latent and the shared rope key, a norm on
the latent only, `kv_b_proj` expanding to per-head keys and values with
`v_head_dim != qk_head_dim`, the interleaved rope convention, the YaRN mscale
folded into the attention scale, and a decode cache that stores latents rather
than expanded keys and values.

Unlocks: DeepSeek V2/V3, Kimi K2/K2.5/K3, GLM-4.7-Flash, and the attention half
of GLM-5.2/5.3 and DeepSeek V3.2.

Parity test: `DeepseekV3Attention` at fp32 on CPU with `q_lora_rank=64`,
`kv_lora_rank=32`, 4 heads, both `rope_interleave` settings, plus a decode test
against prefill. The smallest real MLA checkpoint is
`deepseek-ai/DeepSeek-V2-Lite` at 31.4 GB, which needs CPU RAM, so the
checkpoint-level test is a first-two-layers hidden-state comparison on CPU,
marked as slow.

### Step 8: MTP head. Small, once step 1 exists.

`enorm(embeds)` and `hnorm(previous_hidden)` concatenated,
`eh_proj: 2h -> h`, one copy of the family's block, an optional post-norm, then
the tied head (`TF/modeling_layers.py:316-361`). Two naming conventions to
support: DeepSeek's `enorm`/`hnorm`/`eh_proj`/`shared_head` and Qwen's
`mtp.pre_fc_norm_embedding`/`mtp.pre_fc_norm_hidden`/`mtp.fc`.

For Dew this is more interesting as a **training** signal than as speculative
decoding: an extra loss term on the second-next token. That is an objective
change in `src/dew/objectives/lm`, not a backbone change, and it is the one
item in this list that changes what Dew trains rather than what it can load.

Parity test: `MtpLayer` from `transformers` at fp32 on CPU, copied weights,
identical logits. Plus a loss test showing the MTP term is finite, decreases on
a fixed batch, and can be switched off.

### Step 9: sparse-attention indexers. Large, and last.

DSA's lightning indexer is the simplest of the three
(`TF/models/deepseek_v32/modeling_deepseek_v32.py:160-256`): a small qk
projection, relu scores, per-head weights, top-k into a mask. GLM's adds k-pool
compression; Qwen's QSA adds block compression with a tail rule. All three end
by handing a mask to the same attention kernel, which Dew's
`scaled_dot_product_attention` already accepts.

Unlocks nothing that can be tested on this hardware. Do it when a small
DSA checkpoint exists, or when Dew wants to train its own sparse attention.

### Not planned

| Item | Why |
|---|---|
| Hyper-connections (Qwen4-Exp, DeepSeek V4, GLM-5.3-Flash) | Changes the residual contract of every block. No fixture under 328 GB. |
| Lightning attention (MiniMax Text-01/M1) | One family, superseded by the delta-rule family in that lab's own M2. |
| Mamba2 (Nemotron-H) | Dew has an SSM already; a full Mamba2 port is its own project. |
| Chunked attention and NoPE layers (Llama 4) | Gated repo, 109B smallest, no other family needs them. |
| MXFP4 and FP8 loaders | Loader work, worth doing when a checkpoint that needs it is actually being loaded. |
| AltUp, Laurel (Gemma 3n) | One family, one generation old. |

## Diffusion language models with open weights

Verified on the hub on 2026-09-02.

| Model | Params | Disk bf16 | Type | Forward beyond a causal decoder |
|---|---|---|---|---|
| `google/diffusiongemma-26B-A4B-it` | 25.8B | 51.6 GB | `diffusion_gemma` | Causal encoder over the prompt filling a KV cache, then a **bidirectional** decoder over a 256-token canvas that also attends to the cache. Self-conditioning MLP folding the previous step's logits back into the input embeddings. No mask token: the canvas starts as uniform random ids. No timestep embedding: the only time signal is a temperature annealed by step index in the sampler. |
| `GSAI-ML/LLaDA-8B-Base`, `-8B-Instruct`, `LLaDA-1.5` | 8.0B | 16.0 GB | `llada`, remote code | Fully bidirectional attention, no causal mask anywhere. A real **mask token**: `mask_token_id=126336` inside `vocab_size=126464` (`embedding_size` is also 126464), so the mask id is a reserved slot near the top of the vocabulary. 32 layers, `d_model=4096`, 32 heads and 32 kv heads (no GQA), `mlp_hidden_size=12288`, `rope_theta=500000`, `max_sequence_length=4096`, no biases. Masked-diffusion training predicts all masked positions at once. No timestep input in the config. |
| `GSAI-ML/LLaDA-MoE-v2-30B-A3B-Base`, `-Instruct` | not measured | n/a | `llada` family | As LLaDA plus MoE. |
| `inclusionAI/LLaDA-MoE-7B-A1B-Base`, `-Instruct` | 7.36B | 14.7 GB | `llada` | `moe_router_score_function='softmax'`, 64 experts, 8 active, `qk_layernorm=True`, `dense_intermediate_size=8192`, `expert_intermediate_size=1024`. Bidirectional plus mask token as above. |
| `inclusionAI/LLaDA2.0-mini`, `-flash`, `-Uni`, `LLaDA2.1-mini`, `LLaDA2.2-flash` | 16.3B (mini) | 32.5 GB | `llada2_moe` | 256 experts, 8 active, 1 shared, `moe_router_enable_expert_bias=True`, `n_group=8`, `partial_rotary_factor=0.5`, `first_k_dense_replace=1`. So the newest open diffusion LMs use the same sigmoid-plus-bias group-limited router as DeepSeek. |
| `Dream-org/Dream-v0-Base-7B`, `-Instruct-7B`, `Dream-Coder-v0-*`, `DreamOn-v0-7B`, `DreamReasoner-8B` | 7.62B | 15.2 GB | `Dream`, remote code | A Qwen2.5-7B-shaped decoder (hidden 3584, 28 layers, 28 heads, 4 kv heads, rope theta 1e6) with `mask_token_id=151666` and bidirectional attention. |
| `apple/DiffuCoder-7B-Base`, `-Instruct`, `-cpGRPO` | not measured | n/a | remote code | Same shape family as Dream. |

What a diffusion LM forward needs that Dew's `CausalTransformer` does not have:

1. **A bidirectional mask.** Dew's mask helper is
   `causal_attention_mask(query_positions, kv_len, sliding_window)`
   (`causal_attention_mask` in `src/dew/nn/attention.py`) and the block always asks for causal. Two
   modes are needed: fully bidirectional (LLaDA, Dream), and bidirectional over
   a suffix while a cached prefix stays visible (DiffusionGemma). The second is
   the harder one and it is exactly what
   `create_diffusion_decoder_attention_mask` builds
   (`TF/models/diffusion_gemma/modeling_diffusion_gemma.py:1326-1440`).
2. **A mask token in the vocabulary.** LLaDA and Dream reserve an id and train
   the model to fill it. This is a data and objective concern, not a backbone
   one: the objective samples a masking ratio, replaces that fraction of
   positions with the mask id, and computes the loss only on the masked
   positions.
3. **Time conditioning: none of them need it.** This is the useful negative
   result. LLaDA and Dream take no timestep input at all; the noise level is
   implicit in how many positions are masked. DiffusionGemma takes no timestep
   either; it takes the previous step's logits through the self-conditioning MLP
   and anneals temperature in the sampler. So Dew does not need to wire a
   `sigma` or `t` embedding into `CausalTransformer` for any open diffusion LM.
   Dew's diffusion side already has the schedule machinery in
   `src/dew/diffusion/schedules`, and the masking ratio schedule is a discrete
   analogue of it.

[later] with a caveat: of the three pieces, the bidirectional mask is cheap and
useful on its own (it is also what Gemma 3 and Gemma 4's
`use_bidirectional_attention` flags do for image tokens), and the masked-token
objective is a new objective in `src/dew/objectives`, not a backbone change.
The cheapest real target is `inclusionAI/LLaDA-MoE-7B-A1B-Base` at 14.7 GB, and
it needs step 1's MoE anyway. No open diffusion LM fits comfortably on 16 GB in
bf16, so the fixture is a CPU test.

## What each new piece unlocks

| New piece in Dew | Effort | Families it unlocks | Fixture that proves it |
|---|---|---|---|
| Widen the `mlp` and `layer_types` seams | small | prerequisite for everything | existing suite still green |
| MoE FFN + router family | medium | Mixtral, Qwen3-MoE, Qwen3.5/3.6/3.8-MoE, Gemma 4 MoE, and the FFN of every DeepSeek, GLM, Kimi, MiniMax, LLaDA2 | `MixtralSparseMoeBlock` and `DeepseekV3MoE` at fp32 on CPU |
| GatedDeltaNet mixer (+ KDA variant) | large | Qwen3-Next, Qwen3.5/3.6/3.8 dense, Kimi-Linear, GLM-5.3-Flash linear layers | `Qwen/Qwen3.5-0.8B-Base`, 1.7 GB, ungated, on the 4080 |
| Attention sinks, qk-norm variants | small | gpt-oss, MiniMax M2 | `GptOssAttention` at fp32 on CPU |
| Gated attention output + partial rotary + interleaved mRoPE | small | the whole Qwen 3.5/3.6/3.8 dense line, end to end | full logits on `Qwen/Qwen3.5-0.8B-Base` |
| Gemma 4 block: sandwich norms, scale 1.0, v-norm, per-layer head_dim, KV sharing | medium | `gemma-4-E2B`, `-E4B`, and the dense Gemma 4 shape | `google/gemma-4-E2B`, 10.2 GB, ungated |
| PLE (per-layer embeddings) | medium | Gemma 4 E-series exactly, Gemma 3n | full logits on `google/gemma-4-E2B` |
| MLA mixer | large | DeepSeek V2/V3, Kimi K2/K2.5/K3, GLM-4.7-Flash, attention half of GLM-5.x and V3.2 | `DeepseekV3Attention` at fp32 on CPU; `DeepSeek-V2-Lite` two layers on CPU RAM |
| MTP head + second-token loss | small | a training signal every frontier lab uses; loads DeepSeek, GLM, Qwen MTP weights | `MtpLayer` at fp32 on CPU, plus a loss test |
| Bidirectional / suffix-bidirectional mask | small | LLaDA, Dream, DiffusionGemma decoder, Gemma image tokens | mask unit test, then a CPU logits test on `inclusionAI/LLaDA-MoE-7B-A1B-Base` |
| Sparse-attention indexer (DSA, k-pool, QSA) | large | DeepSeek V3.2/V4, GLM-5.2/5.3, Qwen3.8-Flash-Next | none on this hardware today |
| FP8 / MXFP4 dequantising loader | medium | DeepSeek V3.2, GLM-5.3-Flash, MiniMax M2, gpt-oss | round-trip test against a single dequantised tensor |

## Things this note could not establish

- `MiniMaxAI/MiniMax-H3` has no `config.json` at the repo root, so its
  architecture is unknown. `MiniMax-M2.5` and `-M2.7` were listed but not
  fetched.
- `google/gemma-3-*` is gated, so no Gemma 3 config or size was measured here.
  The Gemma 3 field list above comes from the transformers source, which is
  primary, but the checkpoint values were not read.
- `meta-llama/Llama-4-Scout-17B-16E` is gated; no size measured.
- `Qwen/Qwen3.8-Flash-Next` reports 180B parameters over its safetensors. The
  config comment says the PLE embedding is around 45B parameters
  (`TF/models/qwen4_exp/configuration_qwen4_exp.py:97-99`). The split between
  the PLE table and the rest was not computed.
- `moonshotai/Kimi-Linear-*` runs from remote code. Its KDA implementation was
  read through the GLM-5.3-Flash port in transformers, which the source itself
  labels "Kimi-style KDA"
  (`TF/models/glm5_next/modeling_glm5_next.py:585`), not through Moonshot's own
  file.
- Whether `src/dew/nn/ssm.py` is close to Mamba2 was not checked; that decides
  whether Nemotron-H is cheap or expensive.
