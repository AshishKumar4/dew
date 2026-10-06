export function snippet(source, ...names) {
	return names.map((name) => {
		const text = source.split(`# Begin snippet: ${name}\n`)[1]?.split(`# End snippet: ${name}`)[0];
		if (!text) throw new Error(`framework.py: missing snippet ${name}`);
		const lines = text.split('\n');
		const indent = Math.min(...lines.filter((line) => line.trim()).map((line) => line.match(/^ */)[0].length));
		return lines.map((line) => line.slice(indent)).join('\n').trim();
	}).join('\n\n');
}

export function example(source, name) {
	const cell = (...names) => snippet(source, ...names);
	const join = (...parts) => parts.join('\n\n');
	const training = `import jax\nimport optax\nfrom dew import Trainer`;
	const textImports = `import json\nfrom pathlib import Path\nimport numpy as np\nfrom dew.data import ByteTokenizer, Loading, TokenWindows\nfrom dew.nn.backbones import CausalTransformer\nfrom dew.objectives.lm import LMObjective`;
	const output = `out = Path("dew-${name}")\nout.mkdir(exist_ok=True)`;
	const textData = join(textImports, cell('text-fixture'), output, `_, data = text_fixture(out)`);
	const images = `import itertools\nimport numpy as np\nfrom dew import Dataset, Field`;
	const decoder = join(training, textData, cell('decoder'), `model = decoder()\nobjective = LMObjective(model, 64, ema_decay=None)`);
	const pretrained = `import jax.numpy as jnp\nfrom dew.interop import PretrainedDecoder`;
	const serving = join(pretrained, `from dew import MeshSpec\nfrom dew.inference.serving import Server\nfrom dew.sampling import Sampling`,
		`bundle = PretrainedDecoder.load("Qwen/Qwen3-0.6B", dtype=jnp.bfloat16,\n                                param_dtype=jnp.bfloat16, max_seq_len=128, mesh=MeshSpec())`);
	const programs = {
		lm: join(training, textData, cell('lm')),
		jepa: join(training, images, `from dew.objectives.jepa import JepaEncoder, JepaObjective, JepaPredictor, MultiBlockMask`,
			cell('image-fixture'), `data = image_fixture(32)`, cell('jepa')),
		diffusion: join(training, images, `from dew import InputSpec\nfrom dew.nn.backbones import SimpleDiT\nfrom dew.diffusion.presets import Flow\nfrom dew.objectives.diffusion import DiffusionObjective\nfrom dew.sampling import Euler`,
			cell('image-fixture'), `data = image_fixture(8)`, cell('diffusion')),
		pretrained: join(training, pretrained, `import itertools\nfrom pathlib import Path\nimport numpy as np\nfrom dew import Dataset\nfrom dew.objectives.lm import LMObjective`,
			output, cell('pretrained-source', 'finetune-load', 'finetune-data', 'finetune-dataset', 'finetune')),
		serving: join(serving, `prompts = ["The capital of France is", "The capital of Japan is"]`, cell('serving')),
		formats: join(serving, `task = bundle.text_generation(sampling=Sampling(temperature=0))`, cell('int8', 'fp8')),
		mesh: join(training, `from dew import MeshSpec\njax.config.update("jax_num_cpu_devices", 4)`, textData,
			cell('decoder'), `objective = LMObjective(decoder(), 64, ema_decay=None)`, cell('mesh')),
		grpo: join(training, `import itertools\nimport json\nfrom pathlib import Path\nimport numpy as np\nfrom dew import LocalTracker\nfrom dew.data import Loading, Prompts\nfrom dew.nn.backbones import CausalTransformer\nfrom dew.objectives.rl import GRPOObjective, SampledRollout\nfrom dew.sampling import Sampling`,
			output, `tracker = LocalTracker(out / "tracking")\nsteps = 150\nrng = np.random.default_rng(0)\nrecords = tuple(json.dumps({"prompt": rng.integers(0, 13, 4).tolist()}) for _ in range(512))`,
			cell('grpo'), `tracker.close()`),
		reliability: join(cell('determinism'), decoder, `from dew import Checkpoints`, cell('reliability')),
		profile: join(decoder, `from dew import Checkpoints\nimport jax.numpy as jnp`, cell('reliability', 'profile')),
	};
	if (!(name in programs)) throw new Error(`unknown standalone example: ${name}`);
	return programs[name];
}

export function trainingExample(source) {
	return source.replace('import argparse\n', '')
		.replace(/parser = argparse.ArgumentParser\(\)[\s\S]*?steps = parser.parse_args\(\).steps/, 'steps = 1000')
		.replace('tokens = Path(__file__).with_name("tokens")', 'tokens = Path("tokens")');
}
