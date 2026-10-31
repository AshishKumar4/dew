"""Dew's public surface as one deterministic report, golden in docs/api/report.txtar (ledger 3037).

A txtar archive of four sections, each a contract consumers outside the repository bind to:

- `exports`: each name of `dew.__all__`, its kind and signature; an exported class's public methods and
  properties, each with its signature.
- `config`: each dataclass of `dew.config`, its fields with their types and defaults.
- `cli`: each command of `dew` (`dew.cli.main.COMMANDS`, `DATA_COMMANDS` under `dew data`,
  `dew.cli.tpu.COMMANDS` under `dew tpu`), its flags with their types and defaults.
- `records`: each run record type (`dew.telemetry.records.RECORD_TYPES`), its fields with their types, and
  the schema versions the records, generations and kernel reports carry.

An API change moves a line of one section, which tests/test_api_report.py and thresh's published-language
seams (.thresh/thresh.cue) both see.

    python tools/api_report.py           # print the report
    python tools/api_report.py --write   # write it to docs/api/report.txtar
"""
import argparse
import dataclasses
import importlib
import inspect
import pkgutil
import re
import sys
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


def annotation(value) -> str:
    """A type annotation as written: a string annotation as is, else its formatted form."""
    return value if isinstance(value, str) else ADDRESS.sub("", inspect.formatannotation(value))


def signature(obj) -> str:
    """`obj`'s signature, defaults without addresses; "(...)" where it has none to read."""
    try:
        return ADDRESS.sub("", str(inspect.signature(obj)))
    except (TypeError, ValueError):
        return "(...)"


def fields(cls) -> list[str]:
    """A dataclass's fields: `name: type = default`, a default factory by its name, never run (one may read
    the checkout, as a run record's `dew_sha` does); a dataclass it builds is listed on its own."""
    lines = []
    for field in dataclasses.fields(cls):
        line = f"{field.name}: {annotation(field.type)}"
        if field.default is not dataclasses.MISSING:
            line += f" = {shown(field.default)}"
        elif field.default_factory is not dataclasses.MISSING:
            line += f" = {shown(field.default_factory)}()"
        lines.append(line)
    return lines


def members(cls) -> list[str]:
    """An exported class's public methods and properties, each with its signature, those of `object` aside."""
    lines = []
    for name in sorted(vars(cls) | {key: None for base in cls.__mro__[1:-1] for key in vars(base)}):
        if name.startswith("_"):
            continue
        attribute = inspect.getattr_static(cls, name)
        if isinstance(attribute, property):
            lines.append(f"  .{name}: property")
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


def cli() -> list[str]:
    from dew.cli import main, tpu

    lines = []
    groups = (("dew", main.COMMANDS), ("dew data", main.DATA_COMMANDS), ("dew tpu", tpu.COMMANDS))
    for prefix, commands in groups:
        for name, cls in sorted(commands.items()):
            lines.append(f"{prefix} {name}  [{cls.__module__}.{cls.__qualname__}]")
            for line in fields(cls):
                # tyro's flag for a field: its name, underscores as hyphens.
                flag, _, rest = line.partition(":")
                lines.append(f"  --{flag.replace('_', '-')}:{rest}")
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
