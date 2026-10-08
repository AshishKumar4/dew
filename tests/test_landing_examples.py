"""Every editable native landing example defines its inputs in a fresh scope."""

import ast
import builtins
import inspect
import json
import subprocess
import symtable
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_each_native_landing_cell_defines_its_imports_and_data():
    script = """
        import {readFileSync} from 'node:fs';
        import {example, trainingExample} from './site/src/data/framework-examples.mjs';
        const source = readFileSync('site/snippets/framework.py', 'utf8');
        const where = JSON.parse(readFileSync('site/snippets/cells.json', 'utf8'));
        const names = [...where.pool, ...where.colab];
        const cells = Object.fromEntries(names.map(name => [name, example(source, name)]));
        cells.hero = trainingExample(readFileSync('site/src/data/hero.py', 'utf8'));
        process.stdout.write(JSON.stringify(cells));
    """
    cells = json.loads(subprocess.check_output(["node", "--input-type=module", "-e", script], cwd=ROOT))
    assert len(cells) == 13
    # The page and the recorder (site/snippets/cells.py) read the same code out of framework.py.
    sys.path.insert(0, str(ROOT / "site/snippets"))
    import cells as page
    assert {name: page.cell(name) for name in cells if name != "hero"} == {
        name: code for name, code in cells.items() if name != "hero"}
    assert page.training_example((ROOT / "site/src/data/hero.py").read_text()) == cells["hero"]
    for name, code in cells.items():
        compile(code, f"{name}.py", "exec")
        imports = [node for node in ast.parse(code).body if isinstance(node, (ast.Import, ast.ImportFrom))]
        exec(compile(ast.Module(imports, []), f"{name}.py", "exec"), {})
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


def test_live_cells_are_dew_code_whose_calls_reach_the_shared_model_process():
    """The live cells import Dew's own names; in the live kernel model_client stands in for
    them and sends each call to the host's model process."""
    cells = [(ROOT / "site/src/data" / name).read_text() for name in ("text.py", "sampler.py")]
    for cell in cells:
        imports = [node for node in ast.parse(cell).body if isinstance(node, (ast.Import, ast.ImportFrom))]
        assert {node.module for node in imports} <= {"dew.interop", "dew.sampling"}
        exec(compile(ast.Module(imports, []), "cell.py", "exec"), {})
    from dew.interop import PretrainedDecoder
    inspect.signature(PretrainedDecoder.load).bind("model", dtype="float32", max_seq_len=256)
    script = """
import json, sys
sys.path.insert(0, "site/live/container")
import model_client
model_client.MODELS = model_client.Path("site/live/container/text-models")
png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
sent = []
def request(payload):
    sent.append(payload)
    if payload["op"] == "describe":
        return {"repo": payload["repo"], "revision": payload["revision"]}
    return {"text": [" Paris."]} if payload["op"] == "text" else {"pngs": [png]}
model_client.request = request
model_client.install()
for cell in json.loads(sys.argv[1]):
    exec(cell, {})
try:
    import dew.data
except ModuleNotFoundError as error:
    sent.append(str(error))
print(json.dumps(sent))
"""
    finished = subprocess.run([sys.executable, "-c", script, json.dumps(cells)], cwd=ROOT,
                              capture_output=True, text=True, check=True)
    printed, sent = finished.stdout.splitlines()
    sent = json.loads(sent)
    assert printed == " Paris."
    assert [request["op"] if isinstance(request, dict) else "refused" for request in sent] == [
        "text", "describe", "sample", "refused"]
    assert sent[0]["model"] == "HuggingFaceTB/SmolLM2-135M-Instruct" and sent[0]["tokens"] == 24
    assert sent[2]["steps"] == 15
    assert sent[2]["guidance"] == {"scale": 6.0, "interval": [0.15, 0.9], "rescale": 0.0}
    assert "pip install dewml" in sent[3]
