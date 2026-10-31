package thresh

import "list"

// Dew's seams for thresh (github.com/AshishKumar4/thresh, feat/python-extractor; ledger 3037): the places
// where a change is consequential, or probably wrong, whether or not its tests pass. thresh never gates;
// the precheck hands each head's taps to its reviewer, who records whether each one helped.

_src: ["src/dew/*.py", "src/dew/*/*.py", "src/dew/*/*/*.py", "src/dew/*/*/*/*.py", "src/dew/*/*/*/*/*.py"]
_tests: ["tests/*.py"]

// The public surface: tools/api_report.py's golden report, one section a contract.

seams: "api-exports": {
	kind:    "published-language"
	purpose: "dew's exported names and their signatures are what users' programs call"
	paths:   ["docs/api/report.txtar"]
	format:  "golden"
	section: "exports"
}

seams: "api-config": {
	kind:    "published-language"
	purpose: "the config dataclasses' fields and defaults are what users' run files set"
	paths:   ["docs/api/report.txtar"]
	format:  "golden"
	section: "config"
}

seams: "api-cli": {
	kind:    "published-language"
	purpose: "the dew command's flags are what users' scripts pass"
	paths:   ["docs/api/report.txtar"]
	format:  "golden"
	section: "cli"
}

seams: "api-records": {
	kind:    "published-language"
	purpose: "the run records' fields and the schema versions are what saved runs and their readers hold"
	paths:   ["docs/api/report.txtar"]
	format:  "golden"
	section: "records"
}

// The concept audit's owners (docs/design/architecture.md, one owner per fact): each guarded name used
// outside its owner reaches around what the owner establishes.

seams: profiler: {
	kind:    "choke-point"
	purpose: "a capture runs only when a Profiler starts one, and a region is annotated only through it"
	paths:   _src
	type:    "jax.profiler"
	onlyIn:  ["Profiler.*", "capture_options", "region"]
}

seams: programs: {
	kind:    "choke-point"
	purpose: "every outer program Dew compiles is an options.program, which holds its compile options and its cache key"
	paths:   _src
	type:    "jax.jit"
	onlyIn:  ["OuterProgram.*", "program"]
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
	purpose: "a template's matrix instructions are issued by its Family, which holds their layouts and barriers"
	paths:   _src
	type:    "jax.experimental.pallas.mosaic_gpu.*mma*"
	onlyIn:  ["Family.*", "RingFamily.*", "ConsumerFed.*", "Mma.*", "Wgmma.*", "WgmmaFed.*"]
}

seams: "tensor-memory": {
	kind:    "choke-point"
	purpose: "a Blackwell template's tensor-memory instructions are issued by its Family"
	paths:   _src
	type:    "jax.experimental.pallas.mosaic_gpu.tcgen05_*"
	onlyIn:  ["Family.*", "RingFamily.*", "ConsumerFed.*"]
}

seams: "language-model-config": {
	kind:    "shared-kernel"
	purpose: "language_model_config reads a bundle's text config; a direct read skips its fallbacks"
	paths:   _src
	watch:   ["text_config"]
	consult: ["the Models lane, which owns interop/hf_decoders.py's language_model_config"]
}

seams: scalars: {
	kind:    "shared-kernel"
	purpose: "LocalTracker.scalars reads a run's scalar journal; a direct read of its file skips its order and types"
	paths:   list.Concat([_src, _tests])
	watch:   ["scalars.jsonl"]
	consult: ["the Training lane, which owns training/tracker.py's LocalTracker"]
}

// Registries: each entry is a measured or promised choice, declared with why it belongs.

seams: "port-cells": {
	kind:    "declared-set"
	purpose: "each cell routes the delta rule to the fla port on a receipt that shows the port faster"
	paths:   ["src/dew/nn/kernels/recurrence_chunk.py"]
	entries: "PORT_CELLS"
	declared: "(wgmma, forward, a decay a key dimension)": "Kimi delta attention's forward on an H100, the port's 2.03 ms against the template's 3.24"
}

seams: "scan-cells": {
	kind:    "declared-set"
	purpose: "each cell routes the SSD scan to its Pallas port on a receipt that shows the port faster"
	paths:   ["src/dew/nn/kernels/recurrence_chunk.py"]
	entries: "SCAN_CELLS"
	declared: {
		"(wgmma, forward)":     "Nemotron-H's SSD forward on an H100"
		"(wgmma, training)":    "Nemotron-H's SSD training on an H100"
		"(mma.sync, training)": "the SSD's training on an A100"
	}
}

seams: "selective-cells": {
	kind:    "declared-set"
	purpose: "each cell routes the selective scan to XLA where its kernel cannot run or is slower"
	paths:   ["src/dew/nn/kernels/recurrence_chunk.py"]
	entries: "SELECTIVE_CELLS"
	declared: {
		"(mma.sync, forward)":  "strided cp.async on an A100"
		"(mma.sync, training)": "strided cp.async on an A100"
		"(wgmma, forward)":     "the ungated scan on an H100"
		"(wgmma, training)":    "the ungated scan on an H100"
	}
}

seams: "kernel-choices": {
	kind:    "declared-set"
	purpose: "each choice is a kernel a dispatch reports and a run may pin, a name users' configs hold"
	paths:   ["src/dew/nn/kernels/generation.py"]
	entries: "CHOICES"
	declared: {}
}

seams: commands: {
	kind:    "declared-set"
	purpose: "each command is a verb of the dew CLI users' scripts call"
	paths:   ["src/dew/cli/main.py"]
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
	paths:   ["src/dew/cli/main.py"]
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
	declared: {}
}

// Shared kernels: code every lane stands on.

seams: gate: {
	kind:    "shared-kernel"
	purpose: "check 1 and its sizing judge every accuracy site in the suite"
	paths:   ["tests/reference_error.py", "tests/float64_terms.py", "tests/one_draw.py"]
	consult: ["the Gate lane"]
}

seams: suite: {
	kind:    "shared-kernel"
	purpose: "the suite's fixtures and lane environment shape every test's run"
	paths:   ["tests/conftest.py", "tests/lane_environment.py"]
	consult: ["the queue, which runs every head's suite"]
}

seams: "kernel-targets": {
	kind:    "shared-kernel"
	purpose: "the GPU targets and routes decide which kernel every model runs"
	paths:   ["src/dew/nn/kernels/generation.py"]
	consult: ["the Attention, GEMM and Recurrence lanes"]
}

seams: dependencies: {
	kind:    "shared-kernel"
	purpose: "the declared dependencies, from which every lock is built, are every lane's environment"
	paths:   ["pyproject.toml"]
	consult: ["the queue, which builds every environment"]
}
