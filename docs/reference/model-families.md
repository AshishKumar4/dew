# Decoder and vision families

This reference lists family-specific translation code. It does not certify that every checkpoint bearing a family name has been loaded or trained. Select a configuration supported by the translator, and validate the actual weights and deployment you intend to use.

See [training language models](../concepts/language_models.md) for the workflow and [capabilities and limitations](support.md) for validation levels.

## Text decoders

| Family | Model type handled by the translator | Important behavior |
|---|---|---|
| Llama 2/3/3.1 | `llama` | Grouped-query attention and supported rotary scaling forms |
| Mistral | `mistral` | Sliding-window attention |
| Mixtral | `mixtral` | Sparse experts with softmax routing |
| Qwen 2 | `qwen2` | Q/K/V projection biases with a bias-free output projection |
| Qwen 3 | `qwen3` | Q/K normalization and supported sliding layers |
| Qwen3-MoE | `qwen3_moe` | Routed experts and layer-selection settings |
| Gemma 1/2 | `gemma`, `gemma2` | Family-specific activation, normalization, windows, and softcapping |
| Gemma 3 text | `gemma3_text` | Text path; some released rotary configurations are rejected |
| Gemma 3n text | `gemma3n_text` | AltUp, LAuReL, per-layer inputs, activation sparsity, and KV sharing |
| Gemma 4 text | `gemma4_text` | Per-layer inputs, KV sharing, distinct global attention geometry, and routed variants |
| OLMo 3 | `olmo3` | Post-normalization and Q/K normalization; unsupported rotary forms raise |
| Qwen 3.5 text | `qwen3_5_text` | Gated DeltaNet/full attention; the released shared single MTP layer |
| Qwen 3.5 MoE text | `qwen3_5_moe_text` | Normalized top-k softmax routing, sigmoid-gated shared expert and single MTP layer |
| gpt-oss | `gpt_oss` | Attention sinks and clamped expert MLPs; MXFP4 loading unpacks weights |
| DeepSeek V2/V2-Lite | `deepseek_v2` | MLA and family-specific routing/balance loss |
| DeepSeek V3/V3.2 | `deepseek_v3`, `deepseek_v32` | MLA, supported YaRN, grouped routing, shared experts, and the V3.2 indexer |
| Kimi K2 | `kimi_k2` | DeepSeek-derived decoder computation and family metadata |
| GLM 4.5/5 variants | `glm4_moe` | Attention biases, partial rotary, routing, and supported prediction depths |
| Llama 4 text | `llama4_text` | Interleaved rotary behavior, chunked local attention, and experts |
| LLaDA | `llada` | Bidirectional decoder with a mask token |
| Dream | `dream`, `Dream` | Bidirectional Qwen-derived decoder with a mask token |
| Diffusion Gemma text | `diffusion_gemma_text` | Text computation used by the block-diffusion denoiser |

The translator validates supported computational fields. A new checkpoint release can introduce behavior outside this table. Do not discard a rejected field to force it to load: inspect the reference implementation and add the missing computation with parity evidence.

## Vision wrappers

| Wrapper | Vision computation | Current scope |
|---|---|---|
| Gemma 3 | SigLIP tower and projector | Small tower, projector, and wrapper-forward fixtures |
| Llama 4 | Vision rotary, patch ordering, adapter, and projection | Still-image wrapper fixtures; inspect input resolution and token placement requirements |
| Gemma 4 | Position tables, vision rotary, pooler, and multimodal embedder | Fixed square still-image path; ragged/padded/video behavior is restricted |
| Qwen 3.5 | Channel-time patches, full per-frame attention, spatial rotary and merger | Mixed-resolution image/video inputs, cached inference and native MTP training on pinned tiny sources |
| Gemma 3n | MobileNet-v5 encoder and hard/soft vision embeddings | Image-only bundles; audio config and weights are rejected. Tiny forward, gradient, and update comparisons run on CPU |

Audio towers are not implemented. A vision wrapper's successful forward comparison does not automatically qualify its tokenizer, image processor, generation loop, training gradients, or multi-host placement.

## Pinned Qwen3.8 configurations

Qwen3.8 uses existing Qwen3.5 model types. The metadata in
`tests/fixtures/hf/qwen38-source/source.json` pins these two official sources:

| Checkpoint | Revision | Native configuration |
|---|---|---|
| `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | `qwen3_5` conditional model, dense hybrid decoder, images and videos |
| `Qwen/Qwen3.8-2.4T-A95B` | `207bd685a7e3696cfaff12ded7c6a7ea0f88c996` | `qwen3_5_moe_text`, normalized top-k routing and sigmoid-gated shared experts; this published source is text-only |

Every tensor name in their indexes (1,199 dense and 1,609 MoE) maps, including
all shipped MTP tensors. The full weights were not downloaded. Tiny random
checkpoints preserve the source architecture choices, with reduced geometry
and a byte-level Qwen tokenizer carrying the actual source chat template.
`tools/qwen_qualification_reference.py` generates the fixtures using
Transformers 5.16.1. CPU verification covers processor loading from the published split
file layout, logits, native Trainer updates, source export/readback, cached
generation and the default top-k 20 / top-p 0.95 sampling distribution.

The dense fixture mixes images with timestamped five- and three-frame videos
at different spatial resolutions. Temporal padding, frame ordering, M-RoPE,
MTP supervision and a full update agree with the reference at fp32 tolerance
1e-4; the largest observed post-update difference is 8.80e-6. The MoE
fixture's base and MTP-updated logit differences are below 3.5e-6. This does
not establish full-size memory use, bf16 parity, GPU/TPU throughput or
multi-host qualification.

The released single MTP layer shares the target embedding and head. Its
forward composition is pinned to vLLM `qwen3_5_mtp.py` at
`51da0ca66c8065619c79e35dff97aa99aeaf5644`; the CPU fixture composes actual
Transformers decoder, rotary and normalization classes with that published
glue. It does not execute vLLM's distributed kernels or scheduler. More than
one Qwen MTP layer and dedicated MTP embeddings are explicitly rejected;
neither occurs in the two pinned configurations. Prediction layers support
native auxiliary training and independently allocated cached candidate steps;
a speculative accept/reject scheduler is not provided by this qualification.

`Processor.chat(messages, reasoning_effort=..., preserve_thinking=...)`
executes the checkpoint's own template. The pinned templates accept
`low`, `medium` and `xhigh` reasoning effort and reject unknown values.
`Processor(text, videos=..., video_metadata=...)` accepts the actual HF
video inputs and timestamp metadata. Numeric outputs retain the established
row-aligned vision fields: each video frame is an item in the visual table,
with `image_indices` pointing to its projected features. The existing
`input_ids` retain video placeholders and timestamps. Supplied
`mm_token_type_ids` and grid counts must agree with those placeholders.

The model, visual-frame packing, M-RoPE, MTP, loss and sampling paths here
are native NumPy/JAX/Flax. Raw image/video preprocessing and chat preparation
still pass through the existing Transformers processor adapter. That runtime
dependency does not meet the Dew-native preprocessing contract and remains
a separate cutover; the processor comparisons below are reference evidence,
not a claim that native resizing or waveform feature extraction has landed.

The fixture disables frame sampling to exercise odd temporal-patch padding;
production frame selection remains the checkpoint processor's setting.
Transformers 5.16.1's video processor reports that its default per-frame pixel
cap differs from qwen-vl-utils. The tiny comparisons use that pinned
Transformers behavior; they do not claim qwen-vl-utils preprocessing parity.


## Weight formats and memory

Hugging Face loading reads safetensors and configuration files. Some families support export in their HF vocabulary; unsupported combinations raise errors. A tied language head requires consistent embedding/head weights.

FP8 block-scaled weights and MXFP4 weights are dequantized during loading. The resulting memory footprint can exceed the stored checkpoint size. Quantized training through Qwix is a separate feature from checkpoint loading and has separate runtime requirements.

## Reference evidence

Small fixtures use known configurations, inputs, and weights and compare with the corresponding reference library. Their generators record reference versions. These checks cover specific mathematical paths and should be repeated when a reference or translator changes.

There is no blanket full-size, multi-host, or TPU qualification for this list. Record the actual model revision, dependency versions, dtype, backend, memory use, and exercised operations for a deployment claim.
