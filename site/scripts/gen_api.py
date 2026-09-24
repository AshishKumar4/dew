"""Generate the API reference from Dew's source with griffe.

griffe reads `src/dew` statically, so the build imports neither JAX nor Dew.
Every object is documented in full on one page, its home: the page whose
module is the longest prefix of the object's own path. Other pages that export
it list it with a link. The build fails when

- a module declares `__all__` but has no page of its own,
- a module's `__all__` exports a name that no page documents, or
- a docs page, the README, an example, a recipe or a tutorial imports a name
  from Dew that no page documents,

so a new public module has to be placed in GROUPS below before it ships.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import griffe

REPO = Path(__file__).resolve().parents[2]
SITE = REPO / "site"
CONTENT = SITE / "src/content/docs/api"
GENERATED = SITE / "src/generated"
SOURCE = "https://github.com/AshishKumar4/dew/blob/main/"

GROUPS: list[tuple[str, list[str]]] = [
    ("Top level", ["dew"]),
    ("Training", ["dew.training", "dew.training.state", "dew.training.optim", "dew.training.quantization",
                  "dew.training.runtime"]),
    ("Objectives", ["dew.objectives", "dew.objectives.base", "dew.objectives.lm", "dew.objectives.diffusion",
                    "dew.objectives.jepa", "dew.objectives.rl", "dew.objectives.rl.flow",
                    "dew.objectives.rl.harbor", "dew.objectives.rl.scheduler"]),
    ("Data", ["dew.data", "dew.data.chat", "dew.data.images"]),
    ("Models", ["dew.registry", "dew.nn.backbones", "dew.nn.backbones.causal_transformer",
                "dew.nn.backbones.flux", "dew.nn.backbones.qwen_image", "dew.nn.backbones.sd3",
                "dew.nn.diffusion_gemma", "dew.nn.gemma3n",
                "dew.nn.autoencoders", "dew.nn.kernels", "dew.lora"]),
    ("Diffusion and sampling", ["dew.diffusion", "dew.diffusion.process", "dew.diffusion.schedules",
                                "dew.diffusion.schedules.source", "dew.diffusion.schedules.source_grids",
                                "dew.diffusion.presets", "dew.diffusion.discrete", "dew.sampling",
                                "dew.sampling.solvers", "dew.sampling.flow", "dew.sampling.decoding"]),
    ("Inference and interop", ["dew.inference", "dew.inference.tasks", "dew.interop"]),
    ("Conditions and evaluation", ["dew.inputs", "dew.inputs.encoders", "dew.eval", "dew.eval.harness"]),
    ("Configuration", ["dew.config", "dew.config.sweep"]),
    ("Utilities", ["dew.rl", "dew.artifacts", "dew.telemetry.profile"]),
]
PAGES = [module for _, modules in GROUPS for module in modules]

# Places that show users Dew code; every name they import must be documented.
USAGE = ["docs/**/*.md", "README.md", "CONTRIBUTING.md", "examples/*.py", "recipes/**/*.py", "tutorials/*.ipynb"]
USAGE_SKIP = ("docs/research/", "docs/design/")


class Unresolved(Exception):
    pass


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
    return f"api/{module}"


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
    return any(re.search(r"(^|\.)(nn\.)?Module$", str(base)) and "linen" in str(base) or str(base) == "nn.Module"
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


def prose(text: str, linker: Linker, context: str, own: str) -> str:
    """A docstring as Markdown: code spans kept, angle brackets escaped, references linked."""
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
    return f"{SOURCE}{path.as_posix()}#L{obj.lineno}"


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


def registrations() -> dict[str, list[tuple[str, str]]]:
    """Names registered with `@<registry>("name")`, read from the source."""
    found: dict[str, list[tuple[str, str]]] = {}
    for path in sorted((REPO / "src/dew").rglob("*.py")):
        tree = ast.parse(path.read_text())
        module = ".".join(path.relative_to(REPO / "src").with_suffix("").parts).removesuffix(".__init__")
        for node in ast.walk(tree):
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                continue
            for decorator in node.decorator_list:
                if (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name)
                        and decorator.args and isinstance(decorator.args[0], ast.Constant)
                        and isinstance(decorator.args[0].value, str)):
                    found.setdefault(decorator.func.id, []).append((decorator.args[0].value, f"{module}.{node.name}"))
    return found


def render_object(entry: Entry, linker: Linker, context: str, registered: dict[str, list[tuple[str, str]]]) -> str:
    obj = entry.obj
    own = linker.by_path.get(entry.canonical, "")
    lines = [f"## {entry.name}", ""]
    meta = f'<p class="api-meta"><span class="api-kind">{kind_label(obj)}</span>'
    if obj.lineno:
        meta += f' <a class="api-source" href="{source_link(obj)}">source</a>'
    lines += [meta + "</p>", ""]
    if obj.is_class:
        lines += ["```python", class_signature(obj), "```", ""]
    elif obj.is_function:
        lines += ["```python", function_signature(obj, entry.name), "```", ""]
    elif obj.is_attribute:
        annotation = f": {obj.annotation}" if obj.annotation is not None else ""
        value = f" = {obj.value}" if obj.value is not None and len(str(obj.value)) < 200 else ""
        lines += ["```python", f"{entry.name}{annotation}{value}", "```", ""]
    if obj.docstring:
        lines += [prose(obj.docstring.value, linker, context, own), ""]
    if obj.is_attribute and str(obj.annotation or "").startswith("Registry"):
        members = registered.get(entry.name.split(".")[-1], [])
        if members:
            lines += ["| Registered name | Object |", "|---|---|"]
            for name, path in sorted(members):
                url = linker.by_path.get(path)
                target = f"[`{path.rsplit('.', 1)[-1]}`]({url})" if url else f"`{path}`"
                lines.append(f"| `{name}` | {target} |")
            lines.append("")
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
            if name.startswith("_") or member.is_alias or not member.is_function:
                continue
            lines += [f"### {entry.name}.{name}", "", "```python", function_signature(member), "```", ""]
            if member.docstring:
                lines += [prose(member.docstring.value, linker, context, own), ""]
    return "\n".join(lines)


def usage_imports() -> Iterator[tuple[str, str, str]]:
    """(file, module, name) for every `from dew... import name` users are shown."""
    pattern = re.compile(r"from\s+(dew(?:\.\w+)*)\s+import\s+(\([^)]*\)|[^\n#]+)")
    for glob in USAGE:
        for path in sorted(REPO.glob(glob)):
            relative = path.relative_to(REPO).as_posix()
            if relative.startswith(USAGE_SKIP):
                continue
            if path.suffix == ".ipynb":
                notebook = json.loads(path.read_text())
                text = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
            elif path.suffix == ".md":
                # Only code a reader can run: fenced blocks and inline code spans.
                raw = path.read_text()
                fenced = re.findall(r"^(`{3,}|~{3,})[^\n]*\n(.*?)^\1", raw, re.S | re.M)
                inline = re.findall(r"`([^`\n]+)`", re.sub(r"^(`{3,}|~{3,}).*?^\1", "", raw, flags=re.S | re.M))
                text = "\n".join([block for _, block in fenced] + inline)
            else:
                text = path.read_text()
            for match in pattern.finditer(text):
                names = re.sub(r"\s+as\s+\w+", "", match.group(2).strip("() \n"))
                for name in re.split(r"[,\s]+", names):
                    if re.fullmatch(r"[A-Za-z_]\w*", name):
                        yield relative, match.group(1), name


def load() -> griffe.Module:
    """Load `dew` with its namespace subpackages, which griffe's finder skips."""
    root = REPO / "src"
    loader = griffe.GriffeLoader(search_paths=[str(root)])
    package = loader.load("dew")
    for directory in sorted((root / "dew").rglob("*")):
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


def main() -> None:
    package = load()
    registered = registrations()

    def module_of(path: str) -> griffe.Module:
        obj = package if path == "dew" else package[path.removeprefix("dew.")]
        if not obj.is_module:
            raise SystemExit(f"GROUPS lists {path}, which is not a module")
        return obj

    pages = {path: Page(module_of(path), list(public_entries(module_of(path)))) for path in PAGES}

    # Each object's home: the page with the longest module prefix of its path,
    # else the first page that exports it.
    home: dict[str, str] = {}
    for path in PAGES:
        for entry in pages[path].entries:
            prefixes = [page for page in PAGES if entry.canonical.startswith(page + ".")
                        and any(e.canonical == entry.canonical for e in pages[page].entries)]
            home.setdefault(entry.canonical, max(prefixes, key=len) if prefixes else path)

    linker = Linker()
    for path in PAGES:
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
                        if not name.startswith("_") and not member.is_alias and member.is_function:
                            method = f"/{page_slug(path)}/#{anchor(f'{entry.name}.{name}')}"
                            linker.add(f"{entry.canonical}.{name}", method, f"{entry.name}.{name}")
                            linker.add(f"{path}.{entry.name}.{name}", method)

    # Coverage: every module that declares __all__ has a page, and every exported
    # name and every name the docs import is documented.
    problems = []
    for module in iter_modules(package):
        if module.exports is None:
            continue
        if module.path not in pages:
            problems.append(f"{module.path} declares __all__ but has no API page; add it to GROUPS")
        for entry in public_entries(module):
            if entry.canonical not in home:
                problems.append(f"{module.path}.__all__ exports {entry.name} ({entry.canonical}), which no API page documents")
    for file, module_path, name in usage_imports():
        try:
            module = module_of(module_path) if module_path == "dew" else package[module_path.removeprefix("dew.")]
            member = module.members.get(name)
            if member is None:
                problems.append(f"{file}: `from {module_path} import {name}`: {module_path} has no {name}")
                continue
            target = resolve(member)
        except (KeyError, Unresolved) as error:
            problems.append(f"{file}: `from {module_path} import {name}`: {error}")
            continue
        if target.is_module:
            if name in target.members and target.path + "." + name in home:
                continue
            if target.path not in pages:
                problems.append(f"{file}: imports the module {target.path}, which has no API page")
        elif target.path not in home:
            problems.append(f"{file}: imports {module_path}.{name} ({target.path}), which no API page documents")
    if problems:
        print("gen_api: the API reference does not cover what is public:", file=sys.stderr)
        for problem in sorted(set(problems)):
            print(f"  {problem}", file=sys.stderr)
        raise SystemExit(1)

    CONTENT.mkdir(parents=True, exist_ok=True)
    for path in PAGES:
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
                body.append(render_object(entry, linker, path, registered))
        frontmatter = {
            "title": path,
            "description": summary,
            "slug": page_slug(path),
            "editUrl": False,
            "tableOfContents": {"minHeadingLevel": 2, "maxHeadingLevel": 2},
        }
        text = "---\n" + json.dumps(frontmatter, indent=1) + "\n---\n\n" + "\n".join(body).rstrip() + "\n"
        (CONTENT / f"{path}.md").write_text(text)

    # The overview page is docs/reference/core-api.md, which sync-docs wrote;
    # it ends with every module, grouped as in the sidebar.
    overview = SITE / "src/content/docs/api.md"
    if not overview.exists():
        raise SystemExit("gen_api: run sync-docs first; it writes the overview page the module list extends")
    index = ["", "## All modules", "",
             "Each module below has a page generated from its docstrings. A name a module re-exports links to "
             "the page of the module that defines it.", ""]
    for label, modules in GROUPS:
        index += [f"### {label}", "", "| Module | Summary |", "|---|---|"]
        index += [f"| [`{module}`](/{page_slug(module)}/) | {escape(first_sentence(pages[module].module))} |"
                  for module in modules]
        index.append("")
    overview.write_text(overview.read_text().rstrip() + "\n" + "\n".join(index).rstrip() + "\n")

    GENERATED.mkdir(parents=True, exist_ok=True)
    sidebar = [{"label": label, "collapsed": True,
                "items": [{"label": module, "slug": page_slug(module)} for module in modules]}
               for label, modules in GROUPS]
    (GENERATED / "api.json").write_text(json.dumps(sidebar, indent=1) + "\n")
    (GENERATED / "api-index.json").write_text(json.dumps(linker.by_path, indent=1, sort_keys=True) + "\n")
    # What each registry holds, for pages that count or list registered components.
    (GENERATED / "registries.json").write_text(json.dumps(
        {name: [{"name": key, "object": path} for key, path in sorted(entries)] for name, entries in sorted(registered.items())},
        indent=1) + "\n")
    count = sum(1 for path in PAGES for entry in pages[path].entries if home[entry.canonical] == path)
    print(f"gen_api: {len(PAGES)} pages, {count} objects documented")


def iter_modules(module: griffe.Module) -> Iterator[griffe.Module]:
    yield module
    for member in module.members.values():
        if not member.is_alias and member.is_module:
            yield from iter_modules(member)


if __name__ == "__main__":
    main()
