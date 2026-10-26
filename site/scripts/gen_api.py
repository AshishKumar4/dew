"""Generate the API reference from Dew's source with griffe, or another package's: sparx's site runs
this file, at the Dew commit sparx pins, with `--config` naming its own package, pages and paths
(`Package`).

griffe reads `src/dew` statically, so the build imports neither JAX nor Dew.
Every object is documented in full on one page, its home: the page whose
module is the longest prefix of the object's own path. Other pages that export
it list it with a link. The build fails when

- a module declares `__all__` but has no page of its own,
- a module's `__all__` exports a name that no page documents, or
- a docs page, the README, an example, a recipe, a tutorial or a landing
  snippet imports a name from Dew that no page documents, or reaches one
  through a dotted name (`dew.pipeline`, `TextToImage.from_run`) that the
  module or class it starts from does not have,

so a new public module has to be placed in DEW_GROUPS below before it ships.
"""

from __future__ import annotations

import argparse
import ast
import functools
import io
import json
import re
import shlex
import subprocess
import sys
import tempfile
import tokenize
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import griffe

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import lint_scopes as scopes

DEW_GROUPS: list[tuple[str, list[str]]] = [
    ("Top level", ["dew", "dew.batch", "dew.mesh"]),
    ("Training", ["dew.training", "dew.training.budget", "dew.state", "dew.optim", "dew.training.quantization",
                  "dew.training.runtime", "dew.training.distributed", "dew.training.posthoc"]),
    ("Objectives", ["dew.objectives", "dew.objectives.base", "dew.objectives.lm", "dew.objectives.diffusion",
                    "dew.objectives.diffusion.adversarial",
                    "dew.objectives.diffusion.alignment", "dew.objectives.diffusion.consistency",
                    "dew.objectives.diffusion.end_to_end", "dew.objectives.diffusion.few_step",
                    "dew.objectives.diffusion.guidance_distillation",
                    "dew.objectives.jepa", "dew.objectives.supervised", "dew.objectives.rl",
                    "dew.objectives.rl.flow",
                    "dew.objectives.rl.scheduler", "dew.rl.sandbox", "dew.interop.gateway",
                    "dew.interop.harness", "dew.interop.harbor", "dew.interop.openenv"]),
    ("Data", ["dew.data", "dew.data.presets", "dew.data.pipeline", "dew.data.holdout", "dew.data.compile",
              "dew.data.dataset", "dew.data.plan", "dew.data.mixing", "dew.data.cache",
              "dew.data.storage", "dew.data.sinks",
              "dew.data.sources", "dew.data.sources.base", "dew.data.sources.files", "dew.data.sources.memory",
              "dew.data.sources.hub", "dew.data.sources.tfds", "dew.data.sources.pytorch", "dew.data.sources.hf",
              "dew.data.sources.grain", "dew.data.sources.iterator", "dew.data.sources.codec",
              "dew.data.sources.urls", "dew.data.sources.av_utils",
              "dew.data.steps.generic", "dew.data.steps.text", "dew.data.steps.windows", "dew.data.steps.chat",
              "dew.data.chat", "dew.data.images", "dew.data.tokens", "dew.data.text", "dew.data.video",
              "dew.data.audio", "dew.data.preferences", "dew.data.prompts", "dew.data.processors",
              "dew.data.image_codec"]),
    ("Models", ["dew.registry", "dew.quantization", "dew.nn.backbones", "dew.nn.backbones.causal_transformer",
                "dew.nn.backbones.decoder_block", "dew.nn.backbones.layer_plan", "dew.nn.kv_cache",
                "dew.nn.backbones.edm2", "dew.nn.backbones.joint", "dew.nn.backbones.flux",
                "dew.nn.backbones.flux2",
                "dew.nn.backbones.qwen_image", "dew.nn.backbones.sd3", "dew.nn.backbones.wan",
                "dew.nn.backbones.z_image", "dew.nn.mp",
                "dew.nn.diffusion_gemma", "dew.nn.gemma3n", "dew.nn.multimodal", "dew.nn.cross_attention",
                "dew.nn.memory",
                "dew.nn.autoencoders", "dew.nn.autoencoders.pretrained", "dew.nn.categorical", "dew.nn.inputs", "dew.nn.mlp", "dew.nn.protocols", "dew.nn.kernels", "dew.nn.sharding",
                "dew.lora"]),
    ("Diffusion and sampling", ["dew.diffusion", "dew.diffusion.process", "dew.diffusion.schedules",
                                "dew.diffusion.schedules.source", "dew.diffusion.schedules.source_grids",
                                "dew.diffusion.presets", "dew.diffusion.discrete", "dew.sampling",
                                "dew.sampling.solvers", "dew.sampling.flow", "dew.sampling.guidance",
                                "dew.sampling.decoding"]),
    ("Inference and interop", ["dew.inference", "dew.inference.banks", "dew.inference.tasks",
                              "dew.interop", "dew.interop.diffusion_gemma"]),
    ("Decision models", ["dew.decision", "dew.decision.config", "dew.decision.data", "dew.decision.objective",
                         "dew.decision.rows", "dew.decision.layout", "dew.decision.task"]),
    ("Conditions and evaluation", ["dew.inputs", "dew.inputs.encoders", "dew.eval", "dew.eval.harness"]),
    ("Configuration", ["dew.config", "dew.config.lm", "dew.config.jepa", "dew.config.diffusion",
                       "dew.config.sweep"]),
    ("Utilities", ["dew.rl", "dew.artifacts", "dew.distributed", "dew.telemetry.profile",
                   "dew.telemetry.benchmark", "dew.figures"]),
]
# Places that show users Dew code; every Dew name they use must resolve to the
# documented API. A landing snippet file shows only its marked regions.
DEW_USAGE = ["docs/**/*.md", "README.md", "CONTRIBUTING.md", "examples/*.py", "recipes/**/*.py",
         "tutorials/*.ipynb", "site/snippets/*.py", "site/src/data/*.py"]
DEW_USAGE_SKIP = ("docs/research/", "docs/design/")




@dataclass(frozen=True)
class Package:
    """What the reference documents and where it goes. Paths are relative to `repo`."""

    package: str
    repo: Path
    source: str  # the source's blob URL, ending in /
    groups: list[tuple[str, list[str]]]
    content: str = "site/src/content/docs/api"  # the module pages
    slug: str = "api"  # their URL prefix
    overview: str = "site/src/content/docs/api.md"  # the page the module list ends
    overview_title: str = ""  # with a title, the page is written afresh; without, it is extended
    generated: str = "site/src/generated"  # the sidebar and the name index
    usage: list[str] = field(default_factory=list)  # files whose imports must resolve
    declared: str = "site/scripts/gen_api.py"  # the file declaring `groups` at any commit (`groups_at`)
    distribution: str = "dewml"  # the name pip installs the package by (`pinned`)
    usage_skip: tuple[str, ...] = ()
    special: tuple[str, ...] = ()  # private-looking methods a class documents, such as "__call__"
    fence_indented: bool = False  # an indented block after a blank line in a docstring is Python

    def method(self, name: str) -> bool:
        """Whether a class's method `name` is documented."""
        return not name.startswith("_") or name in self.special

    @property
    def pages(self) -> list[str]:
        return [module for _, modules in self.groups for module in modules]

    @classmethod
    def read(cls, path: Path) -> Package:
        """A package from a JSON file of these fields, `repo` relative to the file."""
        fields = json.loads(path.read_text())
        repo = (path.parent / fields.pop("repo", ".")).resolve()
        groups = [(label, list(modules)) for label, modules in fields.pop("groups")]
        return cls(repo=repo, groups=groups, usage_skip=tuple(fields.pop("usage_skip", ())),
                   special=tuple(fields.pop("special", ())), **fields)


DEW = Package("dew", Path(__file__).resolve().parents[2], "https://github.com/AshishKumar4/dew/blob/main/",
              DEW_GROUPS, usage=DEW_USAGE, usage_skip=DEW_USAGE_SKIP)
P = DEW


class Unresolved(Exception):
    pass


def inside(obj: griffe.Object | griffe.Alias) -> griffe.Object | None:
    """The object `obj` names within the package, one alias at a time; None where an alias leads into
    another package (flax's Module, a TypedDict), which griffe has not loaded and whose members this
    does not read. An alias within the package that names nothing, or leads back to itself, raises
    (`Unresolved`)."""
    followed = {obj.path}
    while obj.is_alias:
        path = obj.target_path
        if not f"{path}.".startswith(f"{P.package}."):
            return None
        if path in followed:
            raise Unresolved(f"{obj.path} leads back to {path}")
        followed.add(path)
        try:
            obj = module_at(obj.parent.package, path)
        except KeyError as error:
            raise Unresolved(f"{obj.path} names {path}, which the package does not hold") from error
    return obj


def resolve(obj: griffe.Object | griffe.Alias) -> griffe.Object:
    try:
        return obj.final_target if obj.is_alias else obj
    except (griffe.AliasResolutionError, griffe.CyclicAliasError) as error:
        raise Unresolved(str(error)) from error


def slug(text: str) -> str:
    """GitHub's heading anchor for `text`, as Starlight computes it."""
    return re.sub(r"[^\w\- ]", "", text.lower()).replace(" ", "-")


class Slugger:
    """Anchors for one page's headings in order, deduplicated as github-slugger does: `sample`, then `sample-1`."""

    def __init__(self) -> None:
        self.seen: dict[str, int] = {}

    def __call__(self, text: str) -> str:
        base = result = slug(text)
        while result in self.seen:
            self.seen[base] += 1
            result = f"{base}-{self.seen[base]}"
        self.seen[result] = 0
        return result


def page_slug(module: str) -> str:
    return f"{P.slug}/{module}"


@dataclass
class Entry:
    name: str  # the name this page exports it under
    obj: griffe.Object
    canonical: str


@dataclass
class Page:
    module: griffe.Module
    entries: list[Entry] = field(default_factory=list)


def public_entries(module: griffe.Module) -> Iterator[Entry]:
    if module.exports is not None:
        for name in module.exports:
            if name == "__version__":
                continue
            member = module.members.get(name)
            if member is None:
                raise SystemExit(f"{module.path}.__all__ names {name!r}, which the module does not define or import")
            target = resolve(member)
            # A package whose submodule shares a name with the function it exports.
            if target.is_module and name in target.members:
                target = resolve(target.members[name])
            yield Entry(name, target, target.path)
        return
    for name, member in module.members.items():
        if name.startswith("_") or member.is_alias or member.is_module:
            continue
        if member.is_attribute and not (name[0].isupper() or member.docstring):
            continue
        yield Entry(name, member, member.path)


def is_flax_module(cls: griffe.Class) -> bool:
    return any((re.search(r"(^|\.)(nn\.)?Module$", str(base)) and "linen" in str(base)) or str(base) == "nn.Module"
               for base in cls.bases)


def is_dataclass_like(cls: griffe.Class) -> bool:
    return "dataclass" in cls.labels or is_flax_module(cls) or any(
        "dataclass" in str(decorator.value) for decorator in cls.decorators)


def fields(cls: griffe.Class) -> list[griffe.Attribute]:
    found = []
    for name, member in cls.members.items():
        if member.is_alias or not member.is_attribute or name.startswith("_") or name in ("parent", "name"):
            continue
        if "class-attribute" in member.labels and member.annotation is not None and "ClassVar" not in str(member.annotation):
            found.append(member)
    return found


def parameters(function: griffe.Function) -> list[str]:
    out = []
    star = False
    for parameter in function.parameters:
        if parameter.name in ("self", "cls"):
            continue
        kind = parameter.kind.value if parameter.kind else ""
        text = parameter.name
        if kind == "variadic positional":
            text, star = f"*{parameter.name}", True
        elif kind == "variadic keyword":
            text = f"**{parameter.name}"
        elif kind == "keyword-only" and not star:
            out.append("*")
            star = True
        if parameter.annotation is not None:
            text += f": {parameter.annotation}"
        if parameter.default is not None:
            text += f" = {parameter.default}" if parameter.annotation is not None else f"={parameter.default}"
        out.append(text)
        if kind == "positional-only" and parameter is [p for p in function.parameters if p.kind and p.kind.value == "positional-only"][-1]:
            out.append("/")
    return out


def signature(head: str, params: list[str], returns: str | None = None) -> str:
    tail = f" -> {returns}" if returns else ""
    flat = f"{head}({', '.join(params)}){tail}"
    if len(flat) <= 88:
        return flat
    body = "".join(f"    {param},\n" for param in params)
    return f"{head}(\n{body}){tail}"


def class_signature(cls: griffe.Class) -> str:
    init = cls.members.get("__init__")
    if init is not None and not init.is_alias and init.is_function:
        return signature(f"class {cls.name}", parameters(init))
    if is_dataclass_like(cls):
        params = []
        for attribute in fields(cls):
            text = f"{attribute.name}: {attribute.annotation}"
            if attribute.value is not None:
                text += f" = {attribute.value}"
            params.append(text)
        return signature(f"class {cls.name}", params)
    bases = ", ".join(str(base) for base in cls.bases)
    return f"class {cls.name}({bases})" if bases else f"class {cls.name}"


def function_signature(function: griffe.Function, name: str | None = None) -> str:
    prefix = "async def" if "async" in function.labels else "def"
    returns = str(function.returns) if function.returns is not None else None
    return signature(f"{prefix} {name or function.name}", parameters(function), returns)


def declaration(obj: griffe.Object, name: str | None = None, *, value_limit: int | None = None) -> str:
    if obj.is_class:
        return class_signature(obj)
    if obj.is_function:
        return function_signature(obj, name)
    if obj.is_attribute:
        annotation = f": {obj.annotation}" if obj.annotation is not None else ""
        show_value = obj.value is not None and (value_limit is None or len(str(obj.value)) < value_limit)
        value = f" = {obj.value}" if show_value else ""
        return f"{name or obj.name}{annotation}{value}"
    raise Unresolved(f"{obj.path} has no signature or fields")


def render_references(package: griffe.Module, text: str) -> str:
    """Expand `<!-- api: dotted.name ... -->` with the module pages' source declarations."""
    def expand(match: re.Match) -> str:
        declarations = []
        for path in match[1].split():
            try:
                obj = resolve(module_at(package, path))
            except (KeyError, Unresolved) as error:
                raise Unresolved(f"{path}: {error}") from error
            name = f"{obj.parent.name}.{obj.name}" if obj.parent and obj.parent.is_class else obj.name
            declarations.append(declaration(obj, name))
        return "```text\n" + "\n\n".join(declarations) + "\n```"

    return re.sub(r"^<!-- api: (.+?) -->$", expand, text, flags=re.M)


class Linker:
    """Turns `Name` and `module.Name` code spans in docstrings into links."""

    def __init__(self) -> None:
        self.by_path: dict[str, str] = {}
        self.by_name: dict[str, set[str]] = {}

    def add(self, path: str, url: str, *names: str) -> None:
        self.by_path[path] = url
        for name in names:
            self.by_name.setdefault(name, set()).add(url)

    def url(self, reference: str, context: str) -> str | None:
        reference = reference.strip().rstrip("()")
        if reference in self.by_path:
            return self.by_path[reference]
        if not re.fullmatch(r"[A-Za-z_][\w.]*", reference):
            return None
        urls = self.by_name.get(reference)
        if urls and len(urls) == 1:
            return next(iter(urls))
        # `Class.method` or `module.name`: try the context's own module first.
        candidate = f"{context}.{reference}"
        return self.by_path.get(candidate)


CODE_SPAN = re.compile(r"(`+)(.+?)\1", re.S)


def fenced(text: str) -> list[tuple[bool, str]]:
    """`text` in runs of prose and code: with `P.fence_indented`, an indented block after a blank
    line is code, unindented; otherwise all of it is prose."""
    if not P.fence_indented:
        return [(False, text)]
    runs: list[tuple[bool, list[str]]] = []
    lines = text.split("\n")
    for k, line in enumerate(lines):
        indented = line.startswith("    ")
        starts = indented and line.strip() and (k == 0 or not lines[k - 1].strip())
        code = bool(runs) and runs[-1][0] and (indented or not line.strip())
        if starts or code:
            if not (runs and runs[-1][0]):
                runs.append((True, []))
            runs[-1][1].append(line[4:])
        else:
            if not runs or runs[-1][0]:
                runs.append((False, []))
            runs[-1][1].append(line)
    out = []
    for is_code, chunk in runs:
        while is_code and chunk and not chunk[-1].strip():
            chunk.pop()
        out.append((is_code, "\n".join(chunk)))
    return out


def prose(text: str, linker: Linker, context: str, own: str) -> str:
    """A docstring as Markdown: code spans kept, angle brackets escaped, references linked, and
    indented examples fenced as Python where the package asks for it (`fenced`)."""
    parts = []
    for is_code, chunk in fenced(text):
        parts.append(f"```python\n{chunk}\n```\n" if is_code else spans(chunk, linker, context, own))
    return "".join(parts)


def spans(text: str, linker: Linker, context: str, own: str) -> str:
    text = text.replace("``", "`")
    out, last = [], 0
    for match in CODE_SPAN.finditer(text):
        out.append(escape(text[last:match.start()]))
        code = match.group(2)
        url = linker.url(code, context)
        span = f"`{code}`"
        out.append(f"[{span}]({url})" if url and url != own else span)
        last = match.end()
    out.append(escape(text[last:]))
    return "".join(out)


def escape(text: str) -> str:
    return text.replace("<", "&lt;").replace(">", "&gt;")


def first_sentence(obj: griffe.Object) -> str:
    if not obj.docstring:
        return ""
    paragraph = obj.docstring.value.strip().split("\n\n", 1)[0].replace("\n", " ")
    match = re.match(r"(.+?[.!?])(\s|$)", paragraph)
    return (match.group(1) if match else paragraph).replace("``", "`")


def source_link(obj: griffe.Object) -> str:
    relative = obj.relative_package_filepath if hasattr(obj, "relative_package_filepath") else None
    path = Path("src") / (relative or obj.relative_filepath)
    return f"{P.source}{path.as_posix()}#L{obj.lineno}"


def kind_label(obj: griffe.Object) -> str:
    if obj.is_class:
        if is_flax_module(obj):
            return "Flax module"
        if is_dataclass_like(obj):
            return "dataclass"
        return "class"
    if obj.is_function:
        return "function"
    if obj.is_module:
        return "module"
    return "attribute"


def render_object(entry: Entry, linker: Linker, context: str) -> str:
    obj = entry.obj
    own = linker.by_path.get(entry.canonical, "")
    lines = [f"## {entry.name}", ""]
    meta = f'<p class="api-meta"><span class="api-kind">{kind_label(obj)}</span>'
    if obj.lineno:
        meta += f' <a class="api-source" href="{source_link(obj)}">source</a>'
    lines += [meta + "</p>", ""]
    if obj.is_class or obj.is_function or obj.is_attribute:
        lines += ["```python", declaration(obj, entry.name, value_limit=200), "```", ""]
    if obj.docstring:
        lines += [prose(obj.docstring.value, linker, context, own), ""]
    if obj.is_class:
        documented_fields = [attribute for attribute in fields(obj) if attribute.docstring]
        properties = [member for name, member in obj.members.items()
                      if not name.startswith("_") and not member.is_alias and member.is_attribute
                      and member.docstring and member not in documented_fields]
        if documented_fields or properties:
            lines += ['<dl class="api-fields">']
            for attribute in documented_fields + properties:
                annotation = f": {escape(str(attribute.annotation))}" if attribute.annotation is not None else ""
                lines.append(f"<dt><code>{attribute.name}{annotation}</code></dt>")
                lines.append(f"<dd>\n\n{prose(attribute.docstring.value, linker, context, own)}\n\n</dd>")
            lines += ["</dl>", ""]
        for name, member in obj.members.items():
            if not P.method(name) or member.is_alias or not member.is_function:
                continue
            lines += [f"### {entry.name}.{name}", "", "```python", function_signature(member), "```", ""]
            if member.docstring:
                lines += [prose(member.docstring.value, linker, context, own), ""]
    return "\n".join(lines)


def usage_imports() -> Iterator[tuple[str, str, str]]:
    """(file, module, name) for every `from dew... import name` users are shown,
    read as `unresolved_uses` reads them: the units a file shows
    (`usage_code`), and in Markdown every fenced block and inline code span."""
    for relative, path in usage_files():
        if path.suffix == ".md":
            # Only code a reader can run: fenced blocks and inline code spans.
            raw = path.read_text()
            fenced = re.findall(r"^(`{3,}|~{3,})[^\n]*\n(.*?)^\1", raw, re.S | re.M)
            inline = re.findall(r"`([^`\n]+)`", re.sub(r"^(`{3,}|~{3,}).*?^\1", "", raw, flags=re.S | re.M))
            units = [block for _, block in fenced] + inline
        else:
            units = [code for code, _ in usage_code(relative, path)]
        for unit in units:
            for statement in import_statements(unit):
                for alias in statement.names:
                    yield relative, statement.module, alias.name


def import_statements(code: str) -> Iterator[ast.ImportFrom]:
    """Each `from dew... import ...` in `code`, whole statements where it
    parses and each import line alone where it does not, as a unit cut from
    prose or holding placeholders may not."""
    try:
        trees = [ast.parse(code)]
    except SyntaxError:
        trees = []
        for line in code.splitlines():
            try:
                trees.append(ast.parse(line.strip()))
            except SyntaxError:
                continue
    for tree in trees:
        for node in ast.walk(tree):
            if (isinstance(node, ast.ImportFrom) and node.module
                    and (node.module == P.package or node.module.startswith(P.package + "."))):
                yield node


def usage_files() -> Iterator[tuple[str, Path]]:
    for glob in P.usage:
        for path in sorted(P.repo.glob(glob)):
            relative = path.relative_to(P.repo).as_posix()
            if not relative.startswith(P.usage_skip):
                yield relative, path


def usage_code(relative: str, path: Path) -> Iterator[tuple[str, set[int] | None]]:
    """The Python units a file shows, each with the lines a reader sees (None
    for all): a fenced python block, a notebook code cell, a whole example or
    recipe, or a landing snippet file's marked regions."""
    if path.suffix == ".ipynb":
        for cell in json.loads(path.read_text())["cells"]:
            if cell["cell_type"] == "code":
                # IPython's `%` and `!` lines are not Python. A cell's source is one string or a list
                # of its lines (nbformat's multiline string), which join to the same text.
                yield re.sub(r"^\s*[%!].*", "", "".join(cell["source"]), flags=re.M), None
    elif path.suffix == ".md":
        for block in re.findall(r"^```(?:python|py)[^\n]*\n(.*?)^```", path.read_text(), re.S | re.M):
            yield block, None
    elif relative.startswith("site/snippets/"):
        code = path.read_text()
        shown, inside = set(), False
        for number, line in enumerate(code.splitlines(), 1):
            if "# Begin snippet" in line:
                inside = True
            elif "# End snippet" in line:
                inside = False
            elif inside:
                shown.add(number)
        yield code, shown
    else:
        yield path.read_text(), None


INSTANCE = "<instance>"
"""What a name `name = Class(...)` binds stands for in a scope's table (`held`): this, then the class's
path, which no import's dotted path starts with."""


def reached(package: griffe.Module, dotted: str) -> griffe.Object | None:
    """The Dew object an import's dotted path names, or None for another package's name or one Dew
    does not have; an alias that does not resolve raises `Unresolved`."""
    head, *parts = dotted.split(".")
    if head != P.package:
        return None
    current: griffe.Object = package
    for part in parts:
        member = current.members.get(part)
        if member is None:
            return None
        current = resolve(member)
    return current


def unresolved_uses(package: griffe.Module, home: dict[str, str], pages: dict[str, Page],
                    code: str, shown: set[int] | None, bound: dict[str, str]) -> Iterator[str]:
    """Each dotted use of a Dew name in `code` that does not reach the
    documented API: an attribute a module or class does not have, or a
    module member no page documents. Names come from `import dew...` and
    `from dew... import ...` as Python's scopes read them (tools/lint_scopes.py):
    in this unit or an earlier one of the same file (`bound`, the module
    scope's names, which this updates), except where a scope binds the name
    itself; `name = Class(...)` binds an instance, whose calls must be methods."""
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        yield f"a Python unit does not parse, so none of its names is checked: {error}"
        return
    imports = {id(node): table for node, table in scopes.scoped(tree.body, bound, ())}
    tables = {id(node): table for node, table in
              scopes.scoped(tree.body, bound, (), held=held(tree, imports, package))}
    if tree.body:
        bound.clear()
        bound.update(tables[id(tree.body[0])])

    def chain(node: ast.expr) -> list[str] | None:
        if isinstance(node, ast.Name):
            return [node.id]
        if isinstance(node, ast.Attribute):
            inner = chain(node.value)
            return None if inner is None else [*inner, node.attr]
        return None

    called = {id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    # Only the outermost dotted name counts, so a method and its class are not counted twice; one
    # spelling is checked once a binding.
    inner = {id(node.value) for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    seen = set()
    for node in ast.walk(tree):
        if (not isinstance(node, ast.Attribute) or id(node) in inner
                or (shown is not None and node.lineno not in shown)):
            continue
        path = chain(node)
        target = None if path is None else tables[id(node)].get(path[0])
        if target is None or (target, *path) in seen:
            continue
        seen.add((target, *path))
        try:
            found = reached(package, target.removeprefix(INSTANCE))
            if found is None:
                continue
            problem = unresolved_chain(found, path, home, pages)
            # An instance's members are its class's, and calling one is calling a method.
            if problem is None and target.startswith(INSTANCE) and id(node) in called and len(path) == 2:
                problem = called_attribute(found, path)
        except Unresolved as error:
            problem = f"`{'.'.join(path)}`: {error}"
        if problem:
            yield problem


def held(tree: ast.Module, imports: dict[int, Mapping[str, str]], package: griffe.Module) -> dict[int, str]:
    """Each `name = Class(...)` in `tree` whose class the scope's imports reach as a Dew class: the
    stored name's node identity, to `INSTANCE` and the class's path (`objective =
    DiffusionObjective(...)`), which lint_scopes binds the name to where every binding of it in its
    scope is one such, of one class."""
    instances: dict[int, str] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Call)):
            continue
        try:
            made = reached(package, scopes.resolved(node.value.func, imports[id(node.value.func)]))
        except Unresolved:
            continue
        if made is not None and made.is_class:
            instances[id(node.targets[0])] = INSTANCE + made.path
    return instances


def called_attribute(cls: griffe.Object, path: list[str]) -> str | None:
    """Why calling `path`, an instance's member, leaves the documented API:
    the member is a value an instance holds, not a method, which is what a
    method renamed to make room for a field of the same name leaves behind."""
    found = cls.all_members.get(path[1])
    # A member the class lacks is `unresolved_chain`'s to report; one of another package's, not read.
    member = None if found is None else inside(found)
    if member is None:
        return None
    if member.is_attribute and "instance-attribute" in member.labels and "property" not in member.labels:
        return f"`{'.'.join(path)}(...)`: {path[1]} is a value {cls.path} holds, not a method"
    return None


MEMBERLESS_BASES = ("object", "ABC", "abc.ABC", "Generic", "typing.Generic", "Protocol", "typing.Protocol")
"""Bases outside Dew that give an instance no public attribute of their own."""


def outside_bases(cls: griffe.Class) -> bool:
    """Whether a base of `cls`, or of one of its bases, is defined outside Dew
    and may define attributes (ABC, Generic and Protocol define none)."""
    try:
        resolved = cls.resolved_bases
    except (griffe.AliasResolutionError, griffe.CyclicAliasError):
        return True
    named = [base for base in cls.bases if str(base).split("[")[0] not in MEMBERLESS_BASES]
    return len(resolved) < len(named) or any(outside_bases(base) for base in resolved)


def unresolved_chain(start: griffe.Object, path: list[str], home: dict[str, str],
                     pages: dict[str, Page]) -> str | None:
    """Why `path`, which starts at `start`, leaves the documented API, or None."""
    current = start
    for depth, attribute in enumerate(path[1:], 2):
        used = ".".join(path[:depth])
        if current.is_module:
            if attribute.startswith("__"):
                return None
            member = current.members.get(attribute)
            if member is None or attribute.startswith("_"):
                return f"`{used}`: {current.path} has no public {attribute}"
            current = inside(member)
            # `dew.pipeline` names the function its own module exports.
            if current is not None and current.is_module and f"{current.path}.{attribute}" in home:
                current = inside(current.members[attribute])
            if current is None:
                # Another package's object, whose members this does not read.
                return None
            if not current.is_module and current.path not in home:
                return f"`{used}` ({current.path}) is not documented"
        elif current.is_class:
            if attribute.startswith("_") and not attribute.startswith("__"):
                return f"`{used}`: {attribute} is private to {current.path}"
            members = current.all_members
            if attribute not in members:
                # A base outside Dew (flax, a TypedDict) may define it.
                return None if outside_bases(current) else f"`{used}`: {current.path} has no {attribute}"
            current = inside(members[attribute])
            if current is None:
                # Another package's object, whose members this does not read.
                return None
        else:
            return None
    if current.is_module and current.path not in pages:
        return f"`{'.'.join(path)}` is the module {current.path}, which has no API page"
    return None


def load(root: Path | None = None) -> griffe.Module:
    """Load the package with its namespace subpackages, which griffe's finder skips, from the source under
    `root` (the repository's src/, or a commit's)."""
    root = root or P.repo / "src"
    loader = griffe.GriffeLoader(search_paths=[str(root)])
    package = loader.load(P.package)
    for directory in sorted((root / P.package).rglob("*")):
        if not directory.is_dir() or (directory / "__init__.py").exists() or not any(directory.glob("*.py")):
            continue
        parts = directory.relative_to(root).parts
        parent = package
        for part in parts[1:-1]:
            parent = parent.members[part]
        namespace = griffe.Module(parts[-1], filepath=[directory], parent=parent,
                                  lines_collection=loader.lines_collection,
                                  modules_collection=loader.modules_collection)
        parent.set_member(parts[-1], namespace)
        for file in sorted(directory.glob("*.py")):
            namespace.set_member(file.stem, griffe.visit(
                file.stem, filepath=file, code=file.read_text(), parent=namespace,
                extensions=loader.extensions, docstring_parser=loader.docstring_parser,
                lines_collection=loader.lines_collection, modules_collection=loader.modules_collection))
    loader.resolve_aliases(implicit=False, external=False)
    return package


def module_at(package: griffe.Module, path: str) -> griffe.Object | griffe.Alias:
    """The member at dotted `path`, which starts with the package's own name."""
    return package if path == P.package else package[path.removeprefix(P.package + ".")]


def module_of(package: griffe.Module, path: str) -> griffe.Module:
    obj = module_at(package, path)
    if not obj.is_module:
        raise SystemExit(f"the groups list {path}, which is not a module")
    return obj


def documented(package: griffe.Module, paths: list[str]) -> tuple[dict[str, Page], dict[str, str]]:
    """Each page of the modules `paths` names (`P.pages`, or a pinned commit's), and each documented
    object's home: the page with the longest module prefix of its path, else the first page that exports
    it."""
    pages = {path: Page(module_of(package, path), list(public_entries(module_of(package, path))))
             for path in paths}
    home: dict[str, str] = {}
    for path in paths:
        for entry in pages[path].entries:
            prefixes = [page for page in paths if entry.canonical.startswith(page + ".")
                        and any(e.canonical == entry.canonical for e in pages[page].entries)]
            home.setdefault(entry.canonical, max(prefixes, key=len) if prefixes else path)
    return pages, home


def unresolved_in(file: str, path: Path, package: griffe.Module, pages: dict[str, Page],
                  home: dict[str, str]) -> list[str]:
    """Unresolved code and declaration references, with each Python unit seeing earlier units' bindings."""
    bound: dict[str, str] = {}
    problems = [problem for code, shown in usage_code(file, path)
                for problem in unresolved_uses(package, home, pages, code, shown, bound)]
    if path.suffix == ".md":
        try:
            render_references(package, path.read_text())
        except Unresolved as error:
            problems.append(str(error))
    return problems


def undocumented(package: griffe.Module, pages: dict[str, Page], home: dict[str, str]) -> list[str]:
    """Each module declaring __all__ without a page, and each name it exports that no page documents."""
    problems = []
    for module in iter_modules(package):
        if module.exports is None:
            continue
        if module.path not in pages:
            problems.append(f"{module.path} declares __all__ but has no API page; add it to the groups")
        for entry in public_entries(module):
            if entry.canonical not in home:
                problems.append(f"{module.path}.__all__ exports {entry.name} ({entry.canonical}), which no API page documents")
    return problems


def linked(pages: dict[str, Page], home: dict[str, str]) -> Linker:
    """Every documented name's URL: its page and the anchor of its heading there."""
    linker = Linker()
    for path in P.pages:
        linker.add(path, f"/{page_slug(path)}/")
        # The headings render_object writes, in page order, so the anchors match
        # even when two names differ only in case, like `Sample` and `sample`.
        anchor = Slugger()
        for entry in pages[path].entries:
            if home[entry.canonical] == path:
                url = f"/{page_slug(path)}/#{anchor(entry.name)}"
                linker.add(entry.canonical, url, entry.name)
                linker.add(f"{path}.{entry.name}", url)
                if entry.obj.is_class:
                    for name, member in entry.obj.members.items():
                        if P.method(name) and not member.is_alias and member.is_function:
                            method = f"/{page_slug(path)}/#{anchor(f'{entry.name}.{name}')}"
                            linker.add(f"{entry.canonical}.{name}", method, f"{entry.name}.{name}")
                            linker.add(f"{path}.{entry.name}.{name}", method)
    return linker


def main() -> None:
    global P
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check public API coverage without writing generated files.")
    parser.add_argument("--config", type=Path, help="A JSON file of another package's `Package` fields.")
    args = parser.parse_args()
    if args.config:
        P = Package.read(args.config)
    content, generated = P.repo / P.content, P.repo / P.generated
    package = load()
    pages, home = documented(package, P.pages)

    linker = linked(pages, home)

    # Coverage: every module that declares __all__ has a page, and every exported
    # name and every name the docs import is documented.
    problems = undocumented(package, pages, home)
    # A file shows the API as it is, or as the commit its install pins holds it (`pinned`).
    shown: dict[str, tuple[griffe.Module, dict[str, Page], dict[str, str]]] = {}
    for file, path in usage_files():
        try:
            commit = pinned(path)
            shown[file] = (package, pages, home) if commit is None else at_commit(commit)
        except Unresolved as error:
            problems.append(f"{file}: {error}")
    for file, module_path, name in usage_imports():
        if file not in shown:
            continue
        shown_package, shown_pages, shown_home = shown[file]
        try:
            module = (module_of(shown_package, module_path) if module_path == P.package
                      else module_at(shown_package, module_path))
            member = module.members.get(name)
            if member is None:
                problems.append(f"{file}: `from {module_path} import {name}`: {module_path} has no {name}")
                continue
            target = resolve(member)
        except (KeyError, Unresolved) as error:
            problems.append(f"{file}: `from {module_path} import {name}`: {error}")
            continue
        if target.is_module:
            if name in target.members and target.path + "." + name in shown_home:
                continue
            if target.path not in shown_pages:
                problems.append(f"{file}: imports the module {target.path}, which has no API page")
        elif target.path not in shown_home:
            problems.append(f"{file}: imports {module_path}.{name} ({target.path}), which no API page documents")
    for file, path in usage_files():
        if file in shown:
            problems.extend(f"{file}: {problem}" for problem in unresolved_in(file, path, *shown[file]))
    if problems:
        print("gen_api: the API reference does not cover what is public:", file=sys.stderr)
        for problem in sorted(set(problems)):
            print(f"  {problem}", file=sys.stderr)
        raise SystemExit(1)

    if args.check:
        print(f"gen_api: {len(P.pages)} pages cover the public API")
        return

    content.mkdir(parents=True, exist_ok=True)
    for path in P.pages:
        page = pages[path]
        module = page.module
        summary = first_sentence(module) or f"The `{path}` module."
        body = []
        if module.docstring:
            body += [prose(module.docstring.value, linker, path, f"/{page_slug(path)}/"), ""]
        rows = []
        for entry in page.entries:
            url = linker.by_path.get(entry.canonical, "")
            local = home[entry.canonical] == path
            note = "" if local else f" Documented in [`{home[entry.canonical]}`](/{page_slug(home[entry.canonical])}/)."
            rows.append(f"| [`{entry.name}`]({url}) | {escape(first_sentence(entry.obj)).replace('|', '&#124;')}{note} |")
        if rows:
            body += ["| Name | Summary |", "|---|---|", *rows, ""]
        for entry in page.entries:
            if home[entry.canonical] == path:
                body.append(render_object(entry, linker, path))
        frontmatter = {
            "title": path,
            "description": summary,
            "slug": page_slug(path),
            "editUrl": False,
            "tableOfContents": {"minHeadingLevel": 2, "maxHeadingLevel": 2},
        }
        text = "---\n" + json.dumps(frontmatter, indent=1) + "\n---\n\n" + "\n".join(body).rstrip() + "\n"
        (content / f"{path}.md").write_text(text)

    # The overview page ends with every module, grouped as in the sidebar. Dew's is
    # docs/reference/core-api.md, which sync-docs writes first; a package without one gets a page of its own.
    overview = P.repo / P.overview
    if P.overview_title:
        overview.parent.mkdir(parents=True, exist_ok=True)
        front = json.dumps({"title": P.overview_title, "editUrl": False}, indent=1)
        overview.write_text(f"---\n{front}\n---\n")
    elif not overview.exists():
        raise SystemExit("gen_api: run sync-docs first; it writes the overview page the module list extends")
    index = ["", "## All modules", "",
             "Each module below has a page generated from its docstrings. A name a module re-exports links to "
             "the page of the module that defines it.", ""]
    for label, modules in P.groups:
        index += [f"### {label}", "", "| Module | Summary |", "|---|---|"]
        index += [f"| [`{module}`](/{page_slug(module)}/) | {escape(first_sentence(pages[module].module))} |"
                  for module in modules]
        index.append("")
    overview.write_text(render_references(package, overview.read_text()).rstrip()
                        + "\n" + "\n".join(index).rstrip() + "\n")

    generated.mkdir(parents=True, exist_ok=True)
    sidebar = [{"label": label, "collapsed": True,
                "items": [{"label": module, "slug": page_slug(module)} for module in modules]}
               for label, modules in P.groups]
    (generated / "api.json").write_text(json.dumps(sidebar, indent=1) + "\n")
    (generated / "api-index.json").write_text(json.dumps(linker.by_path, indent=1, sort_keys=True) + "\n")
    count = sum(1 for path in P.pages for entry in pages[path].entries if home[entry.canonical] == path)
    print(f"gen_api: {len(P.pages)} pages, {count} objects documented")


REQUIREMENT = re.compile(r"^(?P<name>[A-Za-z0-9._-]+)(?:\[[^\]]*\])?\s*"
                         r"(?:@\s*git\+(?P<url>[^@\s]+)(?:@(?P<rev>\S+))?)?(?:[=<>!~].*)?$")
"""A requirement a pip install names: a distribution, its extras, and a git repository and revision it
installs from, if any (`dewml[cuda13] @ git+https://github.com/AshishKumar4/dew@<revision>`, with or
without the spaces)."""
COMMIT = re.compile(r"[0-9a-f]{7,40}")


def install_commands(source: str) -> Iterator[list[str]]:
    """The arguments of each `pip install` or `uv pip install` a code cell runs as IPython does: a line
    that starts with `%` or `!`, outside a Python string, with its backslash continuations joined."""
    lines = source.splitlines()
    magic = [index for index, line in enumerate(lines) if line.lstrip().startswith(("%", "!"))]
    # A magic line inside a string literal is the string's text: read the cell as Python with each magic
    # line a statement, and keep those no string token spans.
    python = "\n".join("pass" if index in magic else line for index, line in enumerate(lines))
    tokens = tokenize.generate_tokens(io.StringIO(python).readline)
    strings = [(token.start[0], token.end[0]) for token in tokens if token.type == tokenize.STRING]
    index = 0
    while index < len(lines):
        command, start = lines[index], index
        while command.endswith("\\") and index + 1 < len(lines):
            index += 1
            command = command[:-1] + " " + lines[index]
        index += 1
        if start in magic and not any(first < start + 1 <= last for first, last in strings):
            words = shlex.split(command.lstrip().lstrip("%!"), comments=True)
            if words[:2] == ["pip", "install"] or words[:3] == ["uv", "pip", "install"]:
                yield words


def pinned(path: Path) -> str | None:
    """The commit a notebook's installs of the package (`Package.distribution`) pin it to, which the API
    it shows is that commit's; None where it installs none, or the package as published or at a branch or
    tag. Two installs of different revisions are refused: which API a cell sees would turn on the cell."""
    if path.suffix != ".ipynb":
        return None
    repository = P.source.split("/blob/")[0].removesuffix("/")
    revisions: list[str | None] = []
    for cell in json.loads(path.read_text())["cells"]:
        for words in install_commands("".join(cell["source"])) if cell["cell_type"] == "code" else ():
            for found in filter(None, map(REQUIREMENT.match, words)):
                if found["name"] == P.distribution:
                    url = (found["url"] or "").removesuffix(".git").removesuffix("/")
                    revisions.append(found["rev"] if url == repository and found["rev"] else None)
    if len(set(revisions)) > 1:
        raise Unresolved(f"it installs the package at {len(set(revisions))} revisions: "
                         f"{', '.join(str(revision) for revision in revisions)}")
    return revisions[0] if revisions and revisions[0] and COMMIT.fullmatch(revisions[0]) else None


def groups_at(commit: str) -> list[tuple[str, list[str]]]:
    """The reference's groups as `commit` declared them (`Package.declared`): a JSON file's `groups`, or
    the Python file's `DEW_GROUPS`."""
    text = subprocess.run(["git", "-C", str(P.repo), "show", f"{commit}:{P.declared}"], check=True,
                          stdout=subprocess.PIPE, text=True).stdout
    if P.declared.endswith(".json"):
        return json.loads(text)["groups"]
    for node in ast.parse(text).body:
        if (isinstance(node, ast.Assign | ast.AnnAssign) and node.value is not None
                and any(isinstance(target, ast.Name) and target.id == "DEW_GROUPS"
                        for target in (node.targets if isinstance(node, ast.Assign) else [node.target]))):
            return ast.literal_eval(node.value)
    raise Unresolved(f"{P.declared} declares no DEW_GROUPS at {commit}")


@functools.cache
def at_commit(commit: str) -> tuple[griffe.Module, dict[str, Page], dict[str, str]]:
    """The package's API at `commit` and the reference it documented there: its source read as `load`
    reads the checkout's, and its pages as that commit's groups name them (`groups_at`). A file pinned
    there may use any name that commit documented, whatever pages the reference has now."""
    archive = subprocess.run(["git", "-C", str(P.repo), "archive", commit, "src"], check=True,
                             stdout=subprocess.PIPE).stdout
    with tempfile.TemporaryDirectory() as root:
        subprocess.run(["tar", "-x", "-C", root], input=archive, check=True)
        package = load(Path(root) / "src")
    return package, *documented(package, [module for _, modules in groups_at(commit) for module in modules])


def iter_modules(module: griffe.Module) -> Iterator[griffe.Module]:
    yield module
    for member in module.members.values():
        if not member.is_alias and member.is_module:
            yield from iter_modules(member)


if __name__ == "__main__":
    main()
