/** The landing cell `name`: the code between its markers in site/snippets/framework.py, dedented. */
export function example(source, name) {
	const text = source.split(`# Begin snippet: ${name}\n`)[1]?.split(`# End snippet: ${name}`)[0];
	if (!text) throw new Error(`framework.py: missing snippet ${name}`);
	const lines = text.split('\n');
	const indent = Math.min(...lines.filter((line) => line.trim()).map((line) => line.match(/^ */)[0].length));
	return lines.map((line) => line.slice(indent)).join('\n').trim();
}

export function trainingExample(source) {
	return source.replace('import argparse\n\n', '')
		.replace(/parser = argparse.ArgumentParser\(\)[\s\S]*?steps = parser.parse_args\(\).steps/, 'steps = 1000');
}
