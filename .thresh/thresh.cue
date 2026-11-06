package thresh

import "list"

// Dew's seams for thresh (github.com/AshishKumar4/thresh, feat/python-extractor; ledger 3037): the places
// where a change is consequential, or probably wrong, whether or not its tests pass. thresh never gates;
// a change's reviewer reads its taps and records whether each one helped (`thresh verdict`).

_src: ["src/dew/*.py", "src/dew/*/*.py", "src/dew/*/*/*.py", "src/dew/*/*/*/*.py", "src/dew/*/*/*/*/*.py"]
_tests: ["tests/*.py"]

// The public surface: tools/api_report.py's golden report, one section a contract, each isolated: an edit to
// another section is no touch of it (the cli and records seams tapped every edit to the report: 0 and 1 of 19
// verdicts useful, 2026-11-02).

seams: "api-exports": {
	kind:    "published-language"
	purpose: "dew's exported names and their signatures are what users' programs call"
	paths:   ["docs/api/report.txtar"]
	format:   "golden"
	section:  "exports"
	isolated: true
}

seams: "api-config": {
	kind:    "published-language"
	purpose: "the config dataclasses' fields and defaults are what users' run files set"
	paths:   ["docs/api/report.txtar"]
	format:   "golden"
	section:  "config"
	isolated: true
}

seams: "api-cli": {
	kind:    "published-language"
	purpose: "the dew command's flags are what users' scripts pass"
	paths:   ["docs/api/report.txtar"]
	format:   "golden"
	section:  "cli"
	isolated: true
}

seams: "api-records": {
	kind:    "published-language"
	purpose: "the run records' fields and the schema versions are what saved runs and their readers hold"
	paths:   ["docs/api/report.txtar"]
	format:   "golden"
	section:  "records"
	isolated: true
}

// The concept audit's owners (docs/design/architecture.md, one owner per fact): each guarded name used
// outside its owner reaches around what the owner establishes.

seams: profiler: {
	kind:    "choke-point"
	purpose: "a capture runs only when a Profiler starts one, and a region is annotated only through it"
	paths:   _src
	type:    "jax.profiler"
	onlyIn:  ["Profiler.*", "CaptureOptions.*", "region"]
}

seams: programs: {
	kind:    "choke-point"
	purpose: "every outer program Dew compiles is an options.program, which holds its compile options and its cache key"
	paths:   _src
	type:    "jax.jit"
	// tools/lint_slop.py's COMPILE_POINT, SLOP025's two: the compile point and the nested program.
	onlyIn:  ["OuterProgram.*", "program", "nested", "nested.*"]
}

seams: numerics: {
	kind:    "choke-point"
	purpose: "Numerics owns the matmul precision a program runs at"
	paths:   _src
	type:    "jax.default_matmul_precision"
	onlyIn:  ["Numerics.*", "configured_precision", "at_default_precision", "at_default_precision.*"]
}

seams: tokenizers: {
	kind:    "choke-point"
	purpose: "load_tokenizer pins a tokenizer's revision and resolves its renames once"
	paths:   _src
	type:    "transformers.AutoTokenizer"
	onlyIn:  ["load_tokenizer"]
}

seams: instructions: {
	kind:    "choke-point"
	purpose: "a template's matrix instructions are issued by its bundle (Hopper's and Ampere's Family, Blackwell's Tensor) or its issuer (Ampere's MmaIssuer, Hopper's WgmmaIssuer, Blackwell's Tcgen05 and Signal), or a Blackwell template's product warp, which hold their layouts and barriers"
	paths:   _src
	type:    "jax.experimental.pallas.mosaic_gpu.*mma*"
	onlyIn:  ["Family.*", "RingFamily.*", "ConsumerFed.*", "Mma.*", "Wgmma.*", "WgmmaFed.*", "Tensor.*", "MmaIssuer.*", "WgmmaIssuer.*", "Tcgen05.*", "Signal.*", "_products", "_products.*"]
}

seams: "tensor-commits": {
	kind:    "choke-point"
	purpose: "a Blackwell product's completion is committed by its bundle's Tensor, its issuer's Signal or its template's product warp, the barrier its readers wait on"
	paths:   _src
	type:    "jax.experimental.pallas.mosaic_gpu.tcgen05_commit*"
	onlyIn:  ["Tensor.*", "Signal.*", "_products", "_products.*"]
}

seams: "language-model-config": {
	kind:    "shared-kernel"
	purpose: "language_model_config reads a bundle's text config; a direct read skips its fallbacks"
	paths:   _src
	watch:   ["text_config"]
	// A read it adds; one removed moves a caller onto the owner (11 of 17 verdicts, 2026-11-02).
	added:   true
	consult: ["interop/hf_decoders.py's language_model_config and its maintainers"]
}

seams: scalars: {
	kind:    "shared-kernel"
	purpose: "LocalTracker.scalars reads a run's scalar journal; a direct read of its file skips its order and types"
	paths:   list.Concat([_src, _tests])
	watch:   ["scalars.jsonl"]
	// A read it adds; the 19 taps of row 3024's migration onto LocalTracker were removed reads.
	added:   true
	consult: ["training/tracker.py's LocalTracker and its maintainers"]
}

// Registries: each entry is a measured or promised choice, declared with why it belongs.

seams: "port-cells": {
	kind:    "declared-set"
	purpose: "each cell routes the delta rule to the fla port on a receipt that shows the port faster"
	paths:   ["src/dew/nn/kernels/recurrence_chunk.py"]
	entries: "PORT_CELLS"
	declared: "(wgmma, forward, a decay a key dimension)": "Kimi delta attention's forward on an H100 80GB HBM3, the port's 1.406 ms against the template's 1.624"
}

seams: "scan-cells": {
	kind:    "declared-set"
	purpose: "each cell routes the SSD scan to its Pallas port on a receipt that shows the port faster"
	paths:   ["src/dew/nn/kernels/recurrence_chunk.py"]
	entries: "SCAN_CELLS"
	declared: {
		"(wgmma, training)":    "Nemotron-H's SSD training on an H100"
		"(mma.sync, training)": "the SSD's training on an A100"
	}
}

seams: "grouped-cells": {
	kind:    "declared-set"
	purpose: "each cell routes the grouped GEMM to an old route on a receipt that shows it faster than the template; the list only shrinks"
	paths:   ["src/dew/nn/kernels/grouped_matmul.py"]
	entries: "DECLARED"
	declared: {
		"(mma.sync, dense forward)":          "XLA's dense forward on an A100 (DENSE1)"
		"(mma.sync, dense input_gradient)":   "XLA's dense input gradient on an A100 (DENSE1)"
		"(mma.sync, dense kernel_gradient)":  "XLA's dense kernel gradient on an A100 (DENSE1)"
		"(wgmma, grouped forward)":           "CUTLASS's grouped forward on an H100 (PAIR1-h100)"
		"(wgmma, grouped input_gradient)":    "CUTLASS's grouped input gradient on an H100 (PAIR1-h100)"
		"(wgmma, dense forward)":             "XLA's dense forward on an H100 (DENSE1)"
		"(wgmma, dense input_gradient)":      "XLA's dense input gradient on an H100 (DENSE1)"
		"(wgmma, dense kernel_gradient)":     "XLA's dense kernel gradient on an H100 (DENSE1)"
	}
}

seams: "join-cells": {
	kind:    "declared-set"
	purpose: "each cell runs a join's projections as one GEMM on a device whose receipt shows the join faster"
	paths:   ["src/dew/nn/joined.py"]
	entries: "JOINED_FASTER"
	declared: {
		"(gate_up, sm80)":     "the gate and up projections joined on an A100 (dbf402f36)"
		"(delta_gates, sm80)": "a delta net's gates joined on an A100 (e5198da4e)"
		"(delta_gates, sm90)": "a delta net's gates joined on an H100 (e5198da4e)"
	}
}

seams: "nvfp4-cells": {
	kind:    "declared-set"
	purpose: "each cell runs an NVFP4 product's packed kernel on a receipt that shows it faster than decoding the matrix once a call"
	paths:   ["src/dew/nn/kernels/gemm.py"]
	entries: "NVFP4_CELLS"
	declared: {
		"(sm80, decode)": "the packed kernel at 1-64 rows on an A100 (PACKED_FASTER, W13-a100-4-M-NS-8b-a100)"
	}
}

seams: "norm-cells": {
	kind:    "declared-set"
	purpose: "each cell routes the RMS norm to its fused kernel on a receipt that shows it faster than XLA's"
	paths:   ["src/dew/nn/kernels/norm_swiglu.py"]
	entries: "NORM_CELLS"
	declared: {
		"(sm80, bfloat16)": "the fused norm in bf16 on an A100 (FUSED_FASTER, 14a0df682)"
		"(sm90, bfloat16)": "the fused norm in bf16 on an H100 (FUSED_FASTER, 14a0df682)"
		"(sm80, float16)":  "the fused norm in fp16 on an A100 (FP16_NORM_FASTER, FSNORM16)"
		"(sm90, float16)":  "the fused norm in fp16 on an H100 (FP16_NORM_FASTER, FSNORM16)"
	}
}

seams: "swiglu-cells": {
	kind:    "declared-set"
	purpose: "each cell routes the SiLU product to its fused kernel on a receipt that shows it faster than XLA's"
	paths:   ["src/dew/nn/kernels/norm_swiglu.py"]
	entries: "SWIGLU_CELLS"
	declared: {
		"(sm80, bfloat16)": "the fused SiLU product in bf16 on an A100 (FUSED_FASTER, 14a0df682)"
		"(sm90, bfloat16)": "the fused SiLU product in bf16 on an H100 (FUSED_FASTER, 14a0df682)"
	}
}

seams: "kernel-choices": {
	kind:    "declared-set"
	purpose: "each choice is a kernel a dispatch reports and a run may pin, a name users' configs hold"
	paths:   ["src/dew/nn/kernels/generation.py"]
	entries: "CHOICES"
	declared: {
		attention:          "the attention kernel, a choice a dispatch reports"
		attention_mainloop: "the attention_mainloop kernel, a choice a dispatch reports"
		blockwise_fp8:      "the blockwise_fp8 kernel (a blockwise-scaled FP8 product, row 4518), a choice a dispatch reports"
		decode:             "the decode kernel, a choice a dispatch reports"
		delta_decode:       "the delta_decode kernel, a choice a dispatch reports"
		delta_gates:        "the delta_gates kernel, a choice a dispatch reports"
		gate_up:            "the gate_up kernel, a choice a dispatch reports"
		gated_delta_rule:   "the gated_delta_rule kernel, a choice a dispatch reports"
		gemm_epilogue:      "the gemm_epilogue kernel, a choice a dispatch reports"
		gemm_prologue:      "the gemm_prologue kernel, a choice a dispatch reports"
		grouped_matmul:     "the grouped_matmul kernel, a choice a dispatch reports"
		local_attention:    "the local_attention kernel, a choice a dispatch reports"
		low_rank:           "the low_rank kernel (LoRA's adapted product fused or apart, row 1304), a choice a dispatch reports"
		mxfp8:              "the mxfp8 kernel (an MX product on block-scaled tensor cores or decoded, row 4519), a choice a dispatch reports"
		nvfp4:              "the nvfp4 kernel (an NVFP4 matrix's product decoded in stage or once a call, row 3621), a choice a dispatch reports"
		mla:                "the mla kernel, a choice a dispatch reports"
		narrow_copies:      "the narrow_copies kernel, a choice a dispatch reports"
		paged_decode:       "the paged_decode kernel, a choice a dispatch reports"
		rms_norm:           "the rms_norm kernel, a choice a dispatch reports"
		selective_scan:     "the selective_scan kernel, a choice a dispatch reports"
		short_conv:         "the short_conv kernel, a choice a dispatch reports"
		ssd:                "the ssd kernel, a choice a dispatch reports"
		ssd_decode:         "the ssd_decode kernel, a choice a dispatch reports"
		swiglu:             "the swiglu kernel, a choice a dispatch reports"
	}
}

seams: "phase-flags": {
	kind:    "declared-set"
	purpose: "each flag is one a program compiled ahead may be loaded under unlike its build's, as it computes nothing the program does; one that does would load a program computing otherwise"
	paths:   ["src/dew/training/ahead.py"]
	entries: "PHASE_FLAGS"
	declared: {
		"--xla_gpu_load_autotune_results":                 "which autotune results a build loads, not what they compile to"
		"--xla_gpu_require_complete_aot_autotune_results": "whether a build refuses incomplete autotune results"
		"--xla_gpu_per_fusion_autotune_cache_dir":         "where a build keeps its fusions' autotune results"
	}
}

seams: "dump-flags": {
	kind:    "declared-set"
	purpose: "each flag decides only whether and where XLA dumps, which a run sets for itself (dumps)"
	paths:   ["src/dew/training/ahead.py"]
	entries: "DUMP_FLAGS"
	declared: {
		"--xla_dump":     "XLA's dump directory and kinds"
		"--xla_gpu_dump": "XLA:GPU's dump of its compiled programs"
	}
}

seams: commands: {
	kind:    "declared-set"
	purpose: "each command is a verb of the dew CLI users' scripts call"
	paths:   ["src/dew/_cli/main.py"]
	entries: "COMMANDS"
	declared: {
		export:  "export a run's weights"
		inspect: "print a run's resolved config"
		launch:  "run a command on hosts"
		profile: "profile a run"
		train:   "train a run"
	}
}

seams: "data-commands": {
	kind:    "declared-set"
	purpose: "each command is a verb of dew data users' scripts call"
	paths:   ["src/dew/_cli/main.py"]
	entries: "DATA_COMMANDS"
	declared: {
		export:    "export a dataset"
		provision: "provision a dataset's storage"
		publish:   "publish a dataset"
	}
}

seams: "decoder-families": {
	kind:    "declared-set"
	purpose: "each family is a Hugging Face architecture Dew promises to load"
	paths:   ["src/dew/interop/decoder_families.py"]
	entries: "ENTRIES"
	declared: {
		BLOOM:                "BloomForCausalLM (bloom)"
		DEEPSEEK_V2:          "DeepseekV2ForCausalLM (deepseek_v2)"
		DEEPSEEK_V3:          "DeepseekV3ForCausalLM (deepseek_v3, kimi_k2)"
		DEEPSEEK_V32:         "DeepseekV32ForCausalLM (deepseek_v32)"
		DEEPSEEK_V4:          "DeepseekV4ForCausalLM (deepseek_v4)"
		DEEPSEEK_V41:         "DeepseekV41ForCausalLM (deepseek_v41)"
		DIFFUSION_GEMMA_TEXT: "DiffusionGemmaForBlockDiffusion (diffusion_gemma_text)"
		DORY:                 "DoryForCausalLM (dory)"
		DREAM:                "DreamModel (dream, Dream)"
		FALCON:               "FalconForCausalLM (falcon)"
		GEMMA:                "GemmaForCausalLM (gemma)"
		GEMMA2:               "Gemma2ForCausalLM (gemma2)"
		GEMMA3N_TEXT:         "Gemma3nForCausalLM (gemma3n_text)"
		GEMMA3_TEXT:          "Gemma3ForCausalLM (gemma3_text)"
		GEMMA4_TEXT:          "Gemma4ForCausalLM (gemma4_text)"
		GEMMA4_UNIFIED_TEXT:  "Gemma4UnifiedForCausalLM (gemma4_unified_text)"
		GLM4_MOE:             "Glm4MoeForCausalLM (glm4_moe)"
		GLM5_NEXT_TEXT:       "Glm5NextTextForCausalLM (glm5_next_text)"
		GLM_MOE_DSA:          "GlmMoeDsaForCausalLM (glm_moe_dsa)"
		GPT2:                 "GPT2LMHeadModel (gpt2)"
		GPTJ:                 "GPTJForCausalLM (gptj)"
		GPT_BIGCODE:          "GPTBigCodeForCausalLM (gpt_bigcode)"
		GPT_NEO:              "GPTNeoForCausalLM (gpt_neo)"
		GPT_NEOX:             "GPTNeoXForCausalLM (gpt_neox)"
		GPT_OSS:              "GptOssForCausalLM (gpt_oss)"
		GRANITEMOE:           "GraniteMoeForCausalLM (granitemoe)"
		K2_HORIZON:           "K2HorizonForCausalLM (k2_horizon)"
		KIMI_K25:             "Kimi_K25ForConditionalGeneration (kimi_k25)"
		KIMI_K3:              "KimiK3ForConditionalGeneration (kimi_k3)"
		KIMI_LINEAR:          "KimiLinearForCausalLM (kimi_linear)"
		LFM3:                 "Lfm3ForCausalLM (lfm3)"
		LLADA:                "LLaDAModelLM (llada)"
		LLADA2_MOE:           "LLaDA2MoeModelLM (llada2_moe)"
		LLADA_MOE_V2:         "LLaDAMoEv2ModelLM (llada_moe_v2)"
		LLAMA:                "LlamaForCausalLM (llama)"
		LLAMA4_TEXT:          "Llama4ForCausalLM (llama4_text)"
		MAMBA:                "MambaForCausalLM (mamba)"
		MAMBA2:               "Mamba2ForCausalLM (mamba2)"
		MIMO_V2:              "MiMoV2ForCausalLM (mimo_v2)"
		MIMO_V2_FLASH:        "MiMoV2FlashForCausalLM (mimo_v2_flash)"
		MINIMAX:              "MiniMaxForCausalLM (minimax, minimax_text_01, minimax_m1)"
		MINIMAX_M2:           "MiniMaxM2ForCausalLM (minimax_m2)"
		MINISTRAL:            "MinistralForCausalLM (ministral)"
		MISTRAL:              "MistralForCausalLM (mistral)"
		MIXTRAL:              "MixtralForCausalLM (mixtral)"
		MODERNBERT:           "ModernBertForMaskedLM (modernbert)"
		MUSE_GLIMMER_TEXT:    "MuseGlimmerTextModel (muse_glimmer_text)"
		NEMOTRON_H:           "NemotronHForCausalLM (nemotron_h)"
		OLMO3:                "Olmo3ForCausalLM (olmo3)"
		OPENAI_GPT:           "OpenAIGPTLMHeadModel (openai-gpt)"
		OPT:                  "OPTForCausalLM (opt)"
		OURO:                 "OuroForCausalLM (ouro)"
		PHI:                  "PhiForCausalLM (phi)"
		PHI3:                 "Phi3ForCausalLM (phi3)"
		QWEN2:                "Qwen2ForCausalLM (qwen2)"
		QWEN2_MOE:            "Qwen2MoeForCausalLM (qwen2_moe)"
		QWEN3:                "Qwen3ForCausalLM (qwen3)"
		QWEN3_5_MOE_TEXT:     "Qwen3_5MoeForCausalLM (qwen3_5_moe_text)"
		QWEN3_5_TEXT:         "Qwen3_5ForCausalLM (qwen3_5_text)"
		QWEN3_MOE:            "Qwen3MoeForCausalLM (qwen3_moe)"
		QWEN3_NEXT:           "Qwen3NextForCausalLM (qwen3_next)"
		QWEN3_VL_MOE_TEXT:    "Qwen3VLMoeTextModel (qwen3_vl_moe_text)"
		QWEN4_EXP_TEXT:       "Qwen4ExpForCausalLM (qwen4_exp_text)"
		STARCODER2:           "Starcoder2ForCausalLM (starcoder2)"
	}
}

// Shared kernels: code the whole repository stands on.

seams: gate: {
	kind:    "shared-kernel"
	purpose: "check 1 and its sizing judge every accuracy site in the suite"
	paths:   ["tests/reference_error.py", "tests/float64_terms.py", "tests/one_draw.py"]
	// Every change it makes: a watch on what decides a verdict lost the one useful tap of 8 (a change
	// inside _site_terms, which names none of it on its own lines), 2026-11-02.
	consult: ["the accuracy gate's maintainers (docs/design/testing.md)"]
}

seams: suite: {
	kind:    "shared-kernel"
	purpose: "the suite's fixtures and its environment (tests/lane_environment.py) shape every test's run"
	paths:   ["tests/conftest.py", "tests/lane_environment.py"]
	// What every test computes under, as a change adds it: XLA's flags, JAX's numerics, the mesh, an autouse
	// fixture. The harness's own plumbing (its cache, Hypothesis's directory, collection) is none of it (0 of 5
	// verdicts useful, 2026-11-02).
	watch: ["XLA_FLAGS", "--xla_", "MESH_DEVICES", "jax_enable_x64", "default_matmul_precision", "jax_default_prng",
		"jax_threefry", "autouse"]
	added:   true
	consult: ["the suite's maintainers, as every test runs under them"]
}

seams: "kernel-targets": {
	kind:    "shared-kernel"
	purpose: "the GPU targets and routes decide which kernel every model runs"
	paths:   ["src/dew/nn/kernels/generation.py"]
	consult: ["the attention, GEMM and recurrence kernels' maintainers"]
}

seams: "jax-build": {
	kind:    "shared-kernel"
	purpose: "the JAX build every CI and development install runs on is pinned here, beside the declared dependencies"
	paths:   ["constraints.txt"]
	consult: ["the CI and the JAX fork's maintainers (Platform), who build its wheels for every Python and CUDA"]
}

seams: dependencies: {
	kind:    "shared-kernel"
	purpose: "the declared dependencies, from which every lock is built, are every environment Dew runs in"
	paths:   ["pyproject.toml"]
	consult: ["the CI, which builds every environment from them (tools/environments.py)"]
}

// One owner a sequence of steps.

seams: "near-copies": {
	kind:        "near-copy"
	purpose:     "a sequence of steps written twice drifts apart: one copy is fixed and the other is not; an exact copy is tools/duplicates.py's to refuse, and this notes the near ones"
	paths: ["src/dew/*.py", "src/dew/*/*.py", "src/dew/*/*/*.py", "src/dew/*/*/*/*.py", "src/dew/*/*/*/*/*.py",
		"tests/*.py", "tools/*.py", "tools/*/*.py", "tools/*/*/*.py"]
	containment: 0.9
}
