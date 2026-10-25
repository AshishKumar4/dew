# Supported models

The tables on this page are generated from Dew's source at build time. `Pretrained.load(source)` reads a Hugging Face directory or Hub repository whose `config.json` names one of the `model_type` values below, and `dew.pipeline(source)` wraps the same load in a generation task. `Pretrained.save` writes trained weights back in the source's own layout. Transformers loads the exported directory for the registered causal decoder families; a text-only DiffusionGemma export is refused before anything is written.

A port counts as supported when a test loads the same weights into Dew and into the reference implementation and compares the outputs in float32. For most families that test runs on a small fixture with the release's own configuration and tensor shapes. The "Checked with" column says when a family was also compared at full size against a released checkpoint. Most families have not been run at full size; memory use and throughput at full size are unmeasured for them.

A checkpoint whose type is not listed can still load. A type that computes a registered family's block is verified against transformers at load time. Any other causal LM that Transformers can build loads through Transformers' own PyTorch forward, lowered to JAX by torchax, its attention Dew's (packing included) wherever its layers ask Transformers' attention interface for it, with no sharding rules or cached generation, and `fallback="torchax"` takes that route for any checkpoint. GGUF files, `pytorch_model.bin` repositories and mamba_ssm checkpoints load too. [Train language models](concepts/language_models.md#other-weight-formats) explains each of these paths and what it costs.

## Requested families Dew does not load yet

These families were asked for and are not in the tables above. Each one is tracked until it loads and a test compares it with its reference:

| Family | What is missing |
|---|---|
| Qwen4-Exp (Qwen3.8-Flash-Next) | Its QSA attention, n-gram PLE, and a reader for its config and weights. The gated delta net it uses is in Dew. |
| MiniMax Text-01 and M1 | The lightning-attention layers and a reader. MiniMax M2 loads. |
| MiniMax M3 | The shared expert, MTP and vision-language half, and a reader. |
| LLaDA MoE (GSAI LLaDA-MoE-v2-30B-A3B, inclusionAI LLaDA-MoE-7B-A1B) and LLaDA 2.x | The expert layers and routers in the LLaDA reader, which reads the dense 8B model. |
| Apple DiffuCoder | A pinned load compared with the released model. The Dream reader targets Dream-v0. |
| Kimi K2.5 and K3 vision | The vision towers and projectors. Both load as text decoders. |
| MiniMax M2.5, M2.7 and H3 | Their configurations, to check whether the M2 reader covers them. |
