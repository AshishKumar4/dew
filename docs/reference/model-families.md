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
| Qwen 3.5 text | `qwen3_5_text` | Gated delta-net and full-attention layer combinations |
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

`load_pretrained` builds these wrappers as one native model with the checkpoint's own processor. Each row of the table has a tiny fixture written from the actual Transformers 5.16.1 processor and conditional model, covering the processor call, forward logits, the loss and pixel gradient, an all-parameter SGD update exported back to the source layout and reloaded, and cached greedy generation.

| Wrapper | Media computation | Processor inputs |
|---|---|---|
| Gemma 3 | SigLIP tower and average-pooling projector | Fixed-resolution images, four to 256 soft tokens each |
| Llama 4 | Vision rotary, local and global tiles, pixel-shuffle adapter | Tiled images; the processor normalizes pixels in bfloat16 and the loader widens them exactly |
| Gemma 4 | 2D position tables, vision rotary, clippable linears, position pooler, audio conformer | Padded patch streams with positions, video placeholders, waveforms with one placeholder per encoded frame |
| Qwen 3.5 | Channel-then-time patches, frame-bounded window attention, merger, spatial M-RoPE | Packed patch streams with per-image grids at any resolution the processor emits |
| Gemma 3n | MobileNet-v5 encoder, hard/soft vision and audio embedders, audio conformer | Fixed-resolution images and waveforms with a fixed slot count per clip |

The fixtures are tiny and run on CPU in float32. A passing fixture does not qualify a released checkpoint's memory use, accelerator precision or multi-host placement.

## Weight formats and memory

Hugging Face loading reads safetensors and configuration files. Some families support export in their HF vocabulary; unsupported combinations raise errors. A tied language head requires consistent embedding/head weights.

FP8 block-scaled weights and MXFP4 weights are dequantized during loading. The resulting memory footprint can exceed the stored checkpoint size. Quantized training through Qwix is a separate feature from checkpoint loading and has separate runtime requirements.

## Reference evidence

Small fixtures use known configurations, inputs, and weights and compare with the corresponding reference library. Their generators record reference versions. These checks cover specific mathematical paths and should be repeated when a reference or translator changes.

There is no blanket full-size, multi-host, or TPU qualification for this list. Record the actual model revision, dependency versions, dtype, backend, memory use, and exercised operations for a deployment claim.
