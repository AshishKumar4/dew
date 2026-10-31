"""Dew's public surface as one deterministic report, golden in docs/api/report.txtar (ledger 3037).

A txtar archive of four sections, each a contract consumers outside the repository bind to:

- `exports`: each name of `dew.__all__`, its kind and signature; an exported class's public methods, each
  with its signature, and its properties, each with its getter's type and whether it is set or deleted.
- `config`: each dataclass of `dew.config`, its fields with their types and defaults, and a field that is
  keyword-only or no constructor argument marked so.
- `cli`: each command of `dew` (`dew.cli.main.COMMANDS`, `DATA_COMMANDS` under `dew data`,
  `dew.cli.tpu.COMMANDS` under `dew tpu`), each argument as the command declares it to tyro: its field's
  name, type (aliases resolved) and default, and the markers and spelling tyro is told of; and the
  configuration each group passes tyro (`CONFIG`), which spells them all.
- `records`: each run record type (`dew.telemetry.records.RECORD_TYPES`), its fields as `config`'s, and
  the schema versions the records, generations and kernel reports carry.

An API change moves a line of one section, which tests/test_api_report.py and thresh's published-language
seams (.thresh/thresh.cue) both see.

    python tools/api_report.py           # print the report
    python tools/api_report.py --write   # write it to docs/api/report.txtar
"""
import argparse
import dataclasses
import functools
import importlib
import inspect
import pkgutil
import re
import sys
import types
import typing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "api" / "report.txtar"
ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")
HEADER = ("Dew's public surface (tools/api_report.py). A moved line is an API change: regenerate with\n"
          "`python tools/api_report.py --write` and say why in the commit.\n")


def shown(value) -> str:
    """`value`'s repr without memory addresses, a callable by its qualified name."""
    if inspect.isclass(value) or inspect.isfunction(value) or inspect.isbuiltin(value):
        return f"{getattr(value, '__module__', '?')}.{getattr(value, '__qualname__', repr(value))}"
    return ADDRESS.sub("", repr(value))


@functools.cache
def markers() -> dict[int, str]:
    """tyro's configuration markers, by id: each one's name in `tyro.conf`."""
    import tyro

    return {id(value): name for name, value in vars(tyro.conf).items() if not name.startswith("_")}


def metadata(item) -> str:
    """An `Annotated` type's metadata item: a tyro marker by its name, anything else by its type's."""
    return markers().get(id(item), type(item).__name__)


def annotation(value) -> str:
    """A type annotation as written: a string annotation as is, a union as `A | B`, an `Annotated` type with
    its metadata by their types (a parser's spec holds callables, whose names are no contract), else its
    formatted form."""
    if isinstance(value, str):
        return value
    if value is type(None):
        return "None"
    origin = typing.get_origin(value)
    if origin is typing.Annotated:
        return f"Annotated[{annotation(value.__origin__)}, {', '.join(map(metadata, value.__metadata__))}]"
    if origin in (typing.Union, types.UnionType):
        return " | ".join(annotation(member) for member in typing.get_args(value))
    return ADDRESS.sub("", inspect.formatannotation(value))


def signature(obj) -> str:
    """`obj`'s signature, defaults without addresses; "(...)" where it has none to read."""
    try:
        return ADDRESS.sub("", str(inspect.signature(obj)))
    except (TypeError, ValueError):
        return "(...)"


def default(field) -> str:
    """A field's default as ` = value`, a default factory by its name, never run (one may read the checkout,
    as a run record's `dew_sha` does); "" where it has none."""
    if field.default is not dataclasses.MISSING:
        return f" = {shown(field.default)}"
    if field.default_factory is not dataclasses.MISSING:
        return f" = {shown(field.default_factory)}()"
    return ""


def fields(cls) -> list[str]:
    """A dataclass's fields: `name: type = default`, marked where its constructor takes it by keyword alone
    or not at all; a dataclass it builds is listed on its own."""
    lines = []
    for field in dataclasses.fields(cls):
        line = f"{field.name}: {annotation(field.type)}{default(field)}"
        if not field.init:
            line += "  [no constructor argument]"
        elif field.kw_only is True:
            line += "  [keyword-only]"
        lines.append(line)
    return lines


def returned(getter) -> str:
    """A getter's return annotation as ` -> type`, read without calling it; "" where it has none."""
    try:
        written = inspect.signature(getter).return_annotation
    except (TypeError, ValueError):
        return ""
    return "" if written is inspect.Signature.empty else f" -> {annotation(written)}"


def members(cls) -> list[str]:
    """An exported class's public methods and properties, each with its signature, those of `object` aside."""
    lines = []
    for name in sorted(vars(cls) | {key: None for base in cls.__mro__[1:-1] for key in vars(base)}):
        if name.startswith("_"):
            continue
        attribute = inspect.getattr_static(cls, name)
        if isinstance(attribute, property):
            written = "".join(f" [{verb}]" for verb, method in
                              (("set", attribute.fset), ("deleted", attribute.fdel)) if method is not None)
            lines.append(f"  .{name}: property{returned(attribute.fget)}{written}")
        elif isinstance(attribute, functools.cached_property):
            lines.append(f"  .{name}: cached property{returned(attribute.func)}")
        elif isinstance(attribute, (staticmethod, classmethod)):
            lines.append(f"  .{name}{signature(attribute.__func__)}  [{type(attribute).__name__}]")
        elif inspect.isfunction(attribute):
            lines.append(f"  .{name}{signature(attribute)}")
    return lines


def exports() -> list[str]:
    import dew

    lines = []
    for name in sorted(dew.__all__):
        value = getattr(dew, name)
        if inspect.isclass(value):
            lines.append(f"class {name}{signature(value)}  [{value.__module__}]")
            lines.extend(members(value))
        elif callable(value):
            lines.append(f"def {name}{signature(value)}  [{getattr(value, '__module__', '?')}]")
        elif name == "__version__":  # moves at every release, which the changelog records
            lines.append(f"{name}: {type(value).__name__}")
        else:
            lines.append(f"{name} = {shown(value)}")
    return lines


def config() -> list[str]:
    import dew.config

    modules = [dew.config, *(importlib.import_module(f"dew.config.{info.name}")
                             for info in pkgutil.iter_modules(dew.config.__path__))]
    lines = []
    for module in sorted(modules, key=lambda module: module.__name__):
        for name, cls in sorted(vars(module).items()):
            if inspect.isclass(cls) and dataclasses.is_dataclass(cls) and cls.__module__ == module.__name__:
                lines.append(f"{module.__name__}.{name}")
                lines.extend(f"  {line}" for line in fields(cls))
    return lines


def declared(hint) -> tuple[object, list[str]]:
    """A field's type and what tyro is told of it: each marker of `tyro.conf` by its name, and an argument
    configuration's spelling (`name`, `aliases`, `metavar`), from the `Annotated` layers around the type."""
    notes = []
    while typing.get_origin(hint) is typing.Annotated:
        for item in hint.__metadata__:
            if id(item) in markers():
                notes.append(markers()[id(item)])
            elif type(item).__name__ == "_ArgConfig":
                notes += [f"{key}={value!r}" for key in ("name", "aliases", "metavar")
                          if (value := getattr(item, key)) is not None]
            else:
                notes.append(shown(item))
        hint = hint.__origin__
    return hint, notes


def arguments(cls) -> list[str]:
    """A command's arguments as it declares them to tyro, which spells them (a field `name` the flag
    `--name`, underscores as hyphens, unless it is `Positional` or `arg` names it): each field's name, its
    type with its aliases resolved (`Zone` is `str | None`), its default, and what tyro is told of it."""
    try:
        hints = typing.get_type_hints(cls, include_extras=True)
    except (NameError, TypeError):
        hints = {}
    lines = []
    for field in dataclasses.fields(cls):
        written, notes = declared(hints.get(field.name, field.type))
        marks = "".join(f"  [{note}]" for note in notes)
        lines.append(f"  {field.name}: {annotation(written)}{default(field)}{marks}")
    return lines


def configured(configured_markers) -> str:
    """A command group's tyro configuration, each marker by its name in `tyro.conf`."""
    return ", ".join(markers().get(id(marker), shown(marker)) for marker in configured_markers)


def cli() -> list[str]:
    from dew.cli import main, tpu

    lines = []
    groups = (("dew", main.COMMANDS, main.CONFIG), ("dew data", main.DATA_COMMANDS, main.CONFIG),
              ("dew tpu", tpu.COMMANDS, tpu.CONFIG))
    for prefix, commands, markers in groups:
        lines.append(f"{prefix}  [tyro config: {configured(markers)}]")
        for name, cls in sorted(commands.items()):
            lines.append(f"{prefix} {name}  [{cls.__module__}.{cls.__qualname__}]")
            lines.extend(arguments(cls))
    return lines


def records() -> list[str]:
    from dew.data import generation
    from dew.nn.kernels.templates import report
    from dew.telemetry import records
    from dew.training import rungs

    lines = []
    for cls in records.RECORD_TYPES:
        lines.append(f"{cls.__module__}.{cls.__qualname__}")
        lines.extend(f"  {line}" for line in fields(cls))
    lines += [f"dew.data.generation.GENERATION_SCHEMA = {generation.GENERATION_SCHEMA}",
              f"dew.training.rungs.RECORD_VERSION = {rungs.RECORD_VERSION}",
              f"dew.nn.kernels.templates.report.KERNEL_PROTO_VERSION = {report.KERNEL_PROTO_VERSION}"]
    return lines


SECTIONS = {"exports": exports, "config": config, "cli": cli, "records": records}


def report() -> str:
    """The whole report: its header, then each section as a txtar file."""
    return HEADER + "".join(f"-- {name} --\n" + "\n".join(build()) + "\n" for name, build in SECTIONS.items())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true",
                        help=f"write the report to {REPORT.relative_to(ROOT)}")
    args = parser.parse_args()
    text = report()
    if args.write:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
