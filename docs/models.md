# Supported models

This page is generated from the registries in Dew's source at build time, so it lists what the code loads today. `load_pretrained(source)` reads a Hugging Face directory or Hub repository whose `config.json` names one of the `model_type` values below, and `dew.pipeline(source)` wraps the same load in a generation task. `Pretrained.save` writes trained weights back in the source's own layout, so the directory loads again in transformers.

A port counts as supported when a test loads the same weights into Dew and into the reference implementation and compares the outputs in float32. For most families that test runs on a small fixture with the release's own configuration and tensor shapes. The "Checked with" column says when a family was also compared at full size against a released checkpoint. I have not run most families at full size, so check memory and throughput yourself before a large run.

A checkpoint whose type is not listed can still load. A type that computes the Llama block is verified against transformers at load time, and `fallback="torchax"` runs any transformers causal LM through its own PyTorch code. GGUF files, `pytorch_model.bin` repositories and mamba_ssm checkpoints load too. [Train language models](concepts/language_models.md#other-weight-formats) explains each of these paths and what it costs.
