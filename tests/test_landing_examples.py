"""Every editable native landing example defines its inputs in a fresh scope."""

import builtins
import json
import subprocess
import symtable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_each_native_landing_cell_defines_its_imports_and_data():
    script = """
        import {readFileSync} from 'node:fs';
        import {example, trainingExample} from './site/src/data/framework-examples.mjs';
        const source = readFileSync('site/snippets/framework.py', 'utf8');
        const names = ['lm', 'jepa', 'diffusion', 'pretrained', 'serving', 'formats',
                       'mesh', 'grpo', 'reliability', 'profile'];
        const cells = Object.fromEntries(names.map(name => [name, example(source, name)]));
        cells.hero = trainingExample(readFileSync('site/src/data/hero.py', 'utf8'));
        process.stdout.write(JSON.stringify(cells));
    """
    cells = json.loads(subprocess.check_output(["node", "--input-type=module", "-e", script], cwd=ROOT))
    assert len(cells) == 11
    for name, code in cells.items():
        compile(code, f"{name}.py", "exec")
        table = symtable.symtable(code, f"{name}.py", "exec")
        defined = set(vars(builtins)) | {symbol.get_name() for symbol in table.get_symbols()
                                       if symbol.is_assigned() or symbol.is_imported()}
        scopes = [table]
        while scopes:
            scope = scopes.pop()
            missing = {symbol.get_name() for symbol in scope.get_symbols()
                       if symbol.is_global() and symbol.is_referenced() and symbol.get_name() not in defined}
            assert not missing, (name, missing)
            scopes.extend(scope.get_children())
