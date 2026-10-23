"""An API rename keeps the landing page's recordings: site/scripts/redigest.py, on a copy of the site in a git
repository of its own, writes the renamed programs' digests and leaves every other byte of their recordings
as it was, and writes nothing when a change is more than a declared rename of Dew's API or a recording was
already of other code. The edits are found in the copied programs by parsing them, so the tests follow
whatever API the page uses."""

import ast
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from tool_support import git, load

scopes = load("lint_scopes")

REPO = Path(__file__).resolve().parents[1]
COPIED = ("site/scripts/redigest.py", "site/scripts/gen_api.py", "tools/lint_scopes.py",
          "site/snippets/cells.py", "site/snippets/cells.json",
          "site/snippets/framework.py", "site/src/data/hero.py", "site/src/data/text.py",
          "site/src/data/sampler.py", "site/src/data/capture.json")
DIGEST = re.compile(rb"[0-9a-f]{64}")


@pytest.fixture
def site(tmp_path: Path) -> Path:
    """The page's sources and recordings in a repository whose `recorded` branch is what they ran; the lm
    cell's capture written with CRLF line ends."""
    for relative in COPIED:
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, tmp_path / relative)
    shutil.copytree(REPO / "site/public/examples", tmp_path / "site/public/examples")
    lm = tmp_path / "site/public/examples/framework/lm/capture.json"
    lm.write_bytes(lm.read_bytes().replace(b"\n", b"\r\n"))
    git(tmp_path, "init", "-q", "-b", "recorded")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "recorded")
    return tmp_path


def redigest(root: Path, *renames: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(root / "site/scripts/redigest.py"), "--base", "recorded",
                           *(f"--rename={rename}" for rename in renames)], capture_output=True, text=True,
                          timeout=60)


def api_renamed(root: Path, owner: str, old: str, new: str | None, beside: tuple[str, ...] = ()) -> str:
    """Dew's API at the base holds `owner.old` (owner a dotted class, dew.api.Config), a constructor's
    parameter, and the change renames it `owner.new` (None: keeps it), the classes `beside` in its module
    keeping a parameter `old` of their own; committed: `recorded` the base, `change` the head. The rename to
    declare."""
    module, _, name = owner.rpartition(".")
    parts = module.split(".")
    # The scratch commit's own Dew, which the tool reads at each commit: the module, in its packages.
    for depth in range(1, len(parts) + 1):
        (root / "src" / Path(*parts[:depth])).mkdir(parents=True, exist_ok=True)
        (root / "src" / Path(*parts[:depth], "__init__.py")).touch()
    path = root / "src" / Path(*parts, "__init__.py")
    api = "class {0}:\n    def __init__(self, {1}):\n        self.{1} = {1}\n"
    kept = "".join(api.format(other, old) for other in beside)
    path.write_text(api.format(name, old) + kept)
    git(root, "add", "src")
    git(root, "commit", "-qm", "Dew's API")
    git(root, "checkout", "-q", "-b", "change")
    path.write_text(api.format(name, new or old) + kept)
    git(root, "commit", "-qam", "Dew's API renamed", "--allow-empty")
    return f"{owner}.{old}={owner}.{new or old}"


def api_keywords(source: str) -> list[tuple[str, str]]:
    """Each keyword the program passes a class of Dew's (dew.Trainer, dew.config.OptimConfig) it reaches
    through an import in the call's scope: (the class's dotted path, the keyword), in order."""
    tree = ast.parse(source)
    called = [(scopes.resolved(node.func, table), keyword.arg)
              for node, table in scopes.scoped(tree.body, {}, ()) if isinstance(node, ast.Call)
              for keyword in node.keywords if keyword.arg]
    return list(dict.fromkeys((owner, name) for owner, name in called
                              if owner.startswith("dew.") and owner.rpartition(".")[2][:1].isupper()
                              and not owner.rpartition(".")[0].rpartition(".")[2][:1].isupper()))


def renamed_api_keyword(root: Path, relative: str, owner: str, name: str, new: str) -> None:
    """Keyword `name` renamed `new` in every call of the program's that reaches `owner` through an import."""
    source = (root / relative).read_text()
    tree = ast.parse(source)
    spans = [(keyword, f"{new}={ast.get_source_segment(source, keyword.value)}")
             for node, table in scopes.scoped(tree.body, {}, ())
             if isinstance(node, ast.Call) and scopes.resolved(node.func, table) == owner
             for keyword in node.keywords if keyword.arg == name]
    assert spans
    rewrite(root, relative, spans)


def recordings(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted((root / "site").rglob("*"))
            if path.is_file() and ("public" in path.parts or path.name == "capture.json")}


def program_digest(root: Path, name: str) -> str:
    """The sha256 the page checks program `name`'s recording against, from the copy's own cells.py."""
    code = ("import sys; sys.path.insert(0, 'site/snippets'); import cells; "
            f"print(cells.digest(cells.programs()[{name!r}]))")
    return subprocess.run([sys.executable, "-c", code], cwd=root, check=True, capture_output=True,
                          text=True).stdout.strip()


FRAMEWORK, HERO, TEXT = "site/snippets/framework.py", "site/src/data/hero.py", "site/src/data/text.py"


def programs(root: Path) -> dict[str, str]:
    """The page's programs, from the copy's own cells.py."""
    code = ("import json, sys; sys.path.insert(0, 'site/snippets'); import cells; "
            "print(json.dumps(cells.programs()))")
    return json.loads(subprocess.run([sys.executable, "-c", code], cwd=root, check=True, capture_output=True,
                                     text=True).stdout)


def passed(source: str) -> list[ast.keyword]:
    """The keyword arguments the program passes to its calls, in order."""
    return [keyword for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)
            for keyword in node.keywords if keyword.arg]


def variables(source: str) -> list[str]:
    """The names the program assigns at its top level, in order, each with a later use."""
    tree = ast.parse(source)
    used = [node.id for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)]
    assigned = [target.id for statement in tree.body if isinstance(statement, ast.Assign)
                for target in statement.targets if isinstance(target, ast.Name)]
    return [name for name in dict.fromkeys(assigned) if name in used]


def rewrite(root: Path, relative: str, spans: list[tuple[ast.AST, str]]) -> None:
    """Each node's source in `relative` replaced by its text: the spans the parse gives, back to front."""
    lines = (root / relative).read_bytes().splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    text = b"".join(lines)
    for node, new in sorted(spans, key=lambda span: (span[0].lineno, span[0].col_offset), reverse=True):
        begin = starts[node.lineno - 1] + node.col_offset
        text = text[:begin] + new.encode() + text[starts[node.end_lineno - 1] + node.end_col_offset:]
    (root / relative).write_bytes(text)


def test_a_rename_rewrites_the_digests_of_the_programs_it_reaches_and_keeps_every_other_byte(site):
    """A keyword of Dew's that both the lm cell and the hero pass, renamed as declared in their files: every
    capture naming their programs takes the new digest, and nothing else of it changes, byte for byte."""
    shown = programs(site)
    owner, name = next(pair for pair in api_keywords(shown["lm"]) if pair in api_keywords(shown["hero.py"]))
    declared = api_renamed(site, owner, name, f"{name}_renamed")
    before = recordings(site)
    for relative in (FRAMEWORK, HERO):
        renamed_api_keyword(site, relative, owner, name, f"{name}_renamed")
    assert redigest(site).returncode == 1 and recordings(site) == before, "a rename not declared"
    assert redigest(site, declared).returncode == 0
    after = recordings(site)
    changed = {path for path in before if before[path] != after[path]}
    assert {"site/src/data/capture.json", "site/public/examples/framework/lm/capture.json"} <= changed
    for path in changed:
        assert DIGEST.sub(b"", before[path]) == DIGEST.sub(b"", after[path])
    lm = json.loads(after["site/public/examples/framework/lm/capture.json"])
    assert lm["script_sha256"] == program_digest(site, "lm")
    assert after["site/public/examples/framework/lm/capture.json"].count(b"\r\n") > 1
    hero = json.loads(after["site/src/data/capture.json"])["meta"]
    assert (hero["cell_sha256"], hero["script_sha256"]) == (program_digest(site, "hero"),
                                                            program_digest(site, "hero.py"))
    # Run again, it finds every recording of the program the page shows.
    assert redigest(site, declared).returncode == 0 and recordings(site) == after


def test_a_rename_of_dews_api_keeps_a_recording_only_as_declared_and_where_it_holds(site):
    """A keyword to a class Dew defines, renamed as Dew renames it: kept only under that rename's declaration
    (--rename dew.api.Config.steps=dew.api.Config.num_steps), which the two commits' Dew must hold; the same
    spelling passed to another of Dew's classes (Trainer) or to another callable (dict) is not that rename,
    and keeps no recording."""
    program = "from dew.api import Config, Trainer\nprint(Config({}=3), Trainer({}=3), dict({}=3))\n"
    recorded(site, program.format("steps", "steps", "steps"))
    declared = api_renamed(site, "dew.api.Config", "steps", "num_steps", beside=("Trainer",))
    before = recordings(site)
    for spelled, declaring, kept in ((("steps", "steps"), (), False),
                                     (("num_steps", "steps"), (declared,), False),
                                     (("steps", "num_steps"), (declared,), False),
                                     (("steps", "steps"), (declared,), True)):
        (site / TEXT).write_text(program.format("num_steps", *spelled))
        assert redigest(site, *declaring).returncode == (0 if kept else 1), (spelled, declaring)
        assert (recordings(site) != before) == kept


def an_operation(source: str) -> tuple[ast.Attribute, str]:
    """An attribute the program reads, and another the program reads by a name it does not: swapped, the
    program runs another operation, as np.sin for np.cos."""
    read = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Attribute)]
    names = sorted({node.attr for node in read})
    return read[0], next(name for name in names if name != read[0].attr)


def test_a_program_that_runs_another_operation_keeps_no_recording(site):
    """A recorded program whose call reaches another operation (np.sin for np.cos) prints what its recording
    does not show: refused, with no --rename, with a --rename naming no API of Dew's, and with one Dew's API
    holds but the change does not make, and nothing written."""
    before = recordings(site)
    unchanged = api_renamed(site, "dew.api.Config", "steps", None)
    source = (site / HERO).read_text()
    node, other = an_operation(source)
    rewrite(site, HERO, [(node, f"{ast.get_source_segment(source, node.value)}.{other}")])
    assert programs(site)["hero.py"] != source, "the hero's script is a recorded program"
    absent = "dew.api.Config.steps=dew.api.Config.absent"
    for declared in ((), (f"{node.attr}={other}",), (unchanged,), (absent,)):
        assert redigest(site, *declared).returncode == 1 and recordings(site) == before, declared


def numbered(source: str) -> ast.keyword:
    """The first keyword the program passes a number."""
    return next(keyword for keyword in passed(source)
                if isinstance(keyword.value, ast.Constant) and type(keyword.value.value) in (int, float))


def value_changed(root: Path) -> None:
    keyword = numbered((root / FRAMEWORK).read_text())
    rewrite(root, FRAMEWORK, [(keyword.value, repr(keyword.value.value * 2 + 1))])


def argument_added(root: Path) -> None:
    source = (root / FRAMEWORK).read_text()
    keyword = numbered(source)
    rewrite(root, FRAMEWORK, [(keyword, f"{ast.get_source_segment(source, keyword)}, added=1")])


def variable_renamed(root: Path) -> None:
    """The hero's first variable renamed everywhere the program binds or reads it."""
    name = variables((root / HERO).read_text())[0]
    rewrite(root, HERO, [(node, "renamed_variable") for node in ast.walk(ast.parse((root / HERO).read_text()))
                         if isinstance(node, ast.Name) and node.id == name])


@pytest.mark.parametrize("change", [value_changed, argument_added, variable_renamed],
                         ids=["a value", "an argument", "a variable renamed everywhere"])
def test_a_change_that_is_more_than_a_renaming_writes_nothing(site, change):
    before = recordings(site)
    change(site)
    assert redigest(site).returncode == 1
    assert recordings(site) == before


def recorded(root: Path, source: str) -> None:
    """`source` as the text program the base recorded: text.py, its capture naming its digest."""
    (root / TEXT).write_text(source)
    capture = root / "site/public/examples/text/capture.json"
    named = json.loads(capture.read_bytes())["script_sha256"]
    capture.write_bytes(capture.read_bytes().replace(named.encode(),
                                                     hashlib.sha256(source.encode()).hexdigest().encode()))
    git(root, "commit", "-qam", "a text program recorded")


API = "from dew.api import Config\n"


@pytest.mark.parametrize(("before", "after", "kept"), [
    (f"{API}print(Config(steps=3))\n", f"{API}print(Config(num_steps=3))\n", True),
    ("from dew.api import Config as C\nprint(C(steps=3))\n",
     "from dew.api import Config as C\nprint(C(num_steps=3))\n", True),
    ("import dew.api\nprint(dew.api.Config(steps=3))\n",
     "import dew.api\nprint(dew.api.Config(num_steps=3))\n", True),
    (f"{API}print(Config.steps)\n", f"{API}print(Config.num_steps)\n", True),
    (f"{API}print(Config(steps=3), Config(steps=4))\n", f"{API}print(Config(num_steps=3), Config(steps=4))\n",
     False),
    ("from dew.api import Config as C\ndef run(C):\n    return C(steps=3)\nprint(run(dict))\n",
     "from dew.api import Config as C\ndef run(C):\n    return C(num_steps=3)\nprint(run(dict))\n", False),
    ("from dew.api import Config as C\ndef run(C):\n    return C.steps\nprint(run(C))\n",
     "from dew.api import Config as C\ndef run(C):\n    return C.num_steps\nprint(run(C))\n", False),
    (f"{API}def swap():\n    global Config\n    Config = dict\nswap()\nprint(Config(steps=3))\n",
     f"{API}def swap():\n    global Config\n    Config = dict\nswap()\nprint(Config(num_steps=3))\n", False),
    (f"{API}print(getattr(Config(steps=3), 'steps'))\n",
     f"{API}print(getattr(Config(num_steps=3), 'steps'))\n", False),
    (f"{API}print(Config(steps=3), Config(**{{'steps': 4}}))\n",
     f"{API}print(Config(num_steps=3), Config(**{{'steps': 4}}))\n", False),
    ("print(1)\n", "repr(1)\n", False),
    ("import math as math\nprint(math.pi)\n", "import math as m\nprint(m.pi)\n", False),
    ("import math\nprint(math.pi)\n", "import math as m\nprint(m.pi)\n", False),
    ("from json import dumps\nprint(dumps(1))\n", "from json import dumps as d\nprint(d(1))\n", False),
    ("import os.path\nprint(os.__name__)\n", "import os.path as m\nprint(m.__name__)\n", False),
    ("a, b = 1, 2\nprint(a, b)\n", "b, a = 1, 2\nprint(b, a)\n", False),
    ("x = 1\nprint(f'{x!r}')\n", "x = 1\nprint(f'{x!s}')\n", False),
    ("x = 1\nprint(f'{x=}')\n", "x = 1\nprint(f'{ x = }')\n", False),
    ("print(dict(steps=3))\n", "print(dict(num_steps=3))\n", False),
    ("def f(x):\n    return x\nprint(f(x=1))\n", "def f(y):\n    return y\nprint(f(y=1))\n", False),
    ("def f(x):\n    return x\nf = dict\nprint(f(x=1))\n",
     "def f(y):\n    return y\nf = dict\nprint(f(y=1))\n", False),
    ("def f(x=0, **kw):\n    return x, kw\noptions = {'x': 1}\nprint(f(**options))\n",
     "def f(y=0, **kw):\n    return y, kw\noptions = {'x': 1}\nprint(f(**options))\n", False),
    ("class C:\n    def __init__(self, x):\n        self.x = x\nprint(C(x=1).x)\n",
     "class C:\n    def __init__(self, y):\n        self.x = y\nprint(C(y=1).x)\n", False),
], ids=["a keyword Dew's class takes, renamed as declared", "through the import's alias",
        "through its module",
        "an attribute of Dew's class", "a declared rename left undone at one use",
        "an import's name a parameter shadows", "an attribute through a parameter shadowing the import",
        "an import's name a function rebinds through global", "a name reflection reads from a string",
        "a mapping unpacked into Dew's class", "a builtin for another", "an alias renamed", "an alias added",
        "an imported name given an alias", "an alias added to a dotted import", "two variables swapped",
        "an f-string's conversion", "an f-string's label", "a keyword to a builtin",
        "a function's parameter and its keyword renamed", "a keyword to a name rebound to another callable",
        "a parameter renamed beside a mapping unpacked into it",
        "a class's parameter and its keyword renamed"])
def test_the_programs_parsed_names_decide_what_is_a_rename(site, before, after, kept):
    """A rename is read in the parsed program: only Dew's names it reaches through an import in the scope
    that uses it change, each as the declared rename gives it (dew.api.Config.steps to num_steps), and in
    full; every other name, value and form stays as it was, and nothing reads a renamed name from a string
    or a mapping."""
    recorded(site, before)
    declared = api_renamed(site, "dew.api.Config", "steps", "num_steps")
    was = recordings(site)
    (site / TEXT).write_text(after)
    assert redigest(site, declared).returncode == (0 if kept else 1)
    changed = {path for path in was if was[path] != recordings(site)[path]}
    assert changed == ({"site/public/examples/text/capture.json"} if kept else set())


def a_good_rename_writes_nothing(site: Path, relative: str, broken: str) -> None:
    """A keyword of Dew's that `relative` passes, renamed as declared, a good rename, and the refusal the
    base's capture `broken` holds writes nothing; with `broken` as the site recorded it, the same rename is
    written."""
    recorded_at = git(site, "rev-parse", "HEAD").strip()
    owner, name = api_keywords((site / relative).read_text())[0]
    declared = api_renamed(site, owner, name, f"{name}_renamed")
    before = recordings(site)
    renamed_api_keyword(site, relative, owner, name, f"{name}_renamed")
    assert redigest(site, declared).returncode == 1
    assert recordings(site) == before
    git(site, "checkout", "-q", f"{recorded_at}~1", "--", broken)
    assert redigest(site, declared).returncode == 0 and recordings(site) != before


def test_a_refusal_found_after_a_good_rename_writes_nothing(site):
    """The hero's cell digest written otherwise than the tool would write it: the rename of its script and
    cell is found good, the cell's digest cannot be placed, and neither digest is written."""
    capture = site / "site/src/data/capture.json"
    capture.write_bytes(capture.read_bytes().replace(b'"cell_sha256": "', b'"cell_sha256":"'))
    git(site, "commit", "-qam", "a capture written by hand")
    a_good_rename_writes_nothing(site, HERO, "site/src/data/capture.json")


def test_a_recording_already_of_other_code_writes_nothing(site):
    stale = site / "site/public/examples/framework/jepa/capture.json"
    capture = json.loads(stale.read_text())
    stale.write_text(stale.read_text().replace(capture["script_sha256"], "0" * 64))
    git(site, "commit", "-qam", "a stale recording")
    a_good_rename_writes_nothing(site, FRAMEWORK, "site/public/examples/framework/jepa/capture.json")
