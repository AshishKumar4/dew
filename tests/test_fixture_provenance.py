"""Where every array file under tests/fixtures came from.

tests/fixtures/provenance.json has two parts. `references` holds one record
per fixture (a file, or a directory for every file in it) that claims its
values came from somewhere other than the test reading them. A record names the generating
tool and, for each upstream the numbers come from, its repo and the pinned
release or commit, and how the tool gets them: `runs` names the package the
tool imports and calls, `"runs": "fetched"` says the tool fetches the
upstream's source at that commit (so the commit is in the tool), and
`"transcribed": true` says the tool reimplements the upstream's equations
rather than executing its code. A fixture Dew computed is a Dew regression
golden and says so (`"generator": "dew"`), claiming no upstream.

`inputs` groups the files both sides only read (weights, ids, pixels): the
tool that wrote them, or the pinned repo they were taken from, and the
files. Every .npz, .npy, .safetensors, .pt, .xz and .jaxexport file is in
exactly one of the two parts, so a new array file with no record fails.

A record keyed by a directory claims it: its tool writes the directory whole,
and tests/test_tools.py compares a generator's listing only against a
directory it claims, so no other writer's file may sit in one.
"""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
PROVENANCE = json.loads((FIXTURES / "provenance.json").read_text())
RECORDS = PROVENANCE["references"]
INPUTS = PROVENANCE["inputs"]
ARRAYS = (".npz", ".npy", ".safetensors", ".pt", ".xz", ".jaxexport")


def imports(source: str, package: str) -> bool:
    """Whether `source` imports `package`, statically or through importlib."""
    name = re.escape(package)
    return bool(re.search(rf"^\s*(?:import|from)\s+{name}\b", source, re.M)
                or re.search(rf"import_module\(\s*[\"']{name}[\"'.]", source))


def problems(path: str, record: dict) -> list[str]:
    """Why `record` does not hold for the fixture at `path`, or nothing."""
    found = []
    if not (FIXTURES / path).exists():
        found.append("names no fixture")
    if record.get("generator") == "dew":
        if "sources" in record or "tool" in record:
            found.append("a Dew regression golden names no upstream")
        return found
    tool = ROOT / record.get("tool", "")
    if not record.get("tool") or not tool.is_file():
        return [*found, f"names no generating tool in the repository ({record.get('tool')!r})"]
    source = tool.read_text()
    sources = record.get("sources") or []
    if not sources:
        found.append("names no upstream; a fixture Dew computed says generator: dew")
    for upstream in sources:
        repo, version = upstream.get("repo", ""), upstream.get("version", "")
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo) or repo.lower().endswith("/dew"):
            found.append(f"{repo!r} is not an upstream repo")
        if not version or "unpinned" in version:
            found.append(f"{repo}: no pinned release or commit")
        if upstream.get("transcribed") is True:
            continue
        runs = upstream.get("runs")
        if runs == "fetched":
            commits = re.findall(r"[0-9a-f]{7,40}", version)
            if not any(commit in source for commit in commits):
                found.append(f"{repo}: the tool does not fetch it at {version}")
        elif not runs or not imports(source, runs):
            found.append(f"{repo}: {record['tool']} does not import {runs!r}")
    return found


@pytest.mark.parametrize("path", sorted(RECORDS))
def test_a_fixture_that_names_an_upstream_names_its_pin_and_the_tool_that_runs_it(path):
    assert problems(path, RECORDS[path]) == []


def test_the_rule_refuses_what_it_is_for():
    """Each way a record can overclaim is refused."""
    real = {"tool": "tools/gemma4_tiny_reference.py",
            "sources": [{"repo": "huggingface/transformers", "version": "5.16.1", "runs": "transformers"}]}
    path = "hf/gemma4-ple/logits.npy"
    assert problems(path, real) == []
    assert problems(path, {**real, "tool": "tools/no_such_tool.py"})
    assert problems(path, {**real, "sources": [{**real["sources"][0], "version": "unpinned"}]})
    assert problems(path, {**real, "sources": [{**real["sources"][0], "runs": "diffusers"}]})
    assert problems(path, {**real, "sources": [{**real["sources"][0], "runs": "fetched"}]})
    assert problems(path, {**real, "sources": [{"repo": "AshishKumar4/dew", "version": "1", "runs": "dew"}]})
    assert problems(path, {**real, "sources": []})
    assert problems(path, {"generator": "dew", **real})
    assert problems("no/such/fixture.npz", {"generator": "dew"})
    with pytest.raises(AssertionError):
        test_an_input_group_names_what_wrote_its_files(
            {"tool": "tools/audio_reference.py", "files": ["hf/llama-tiny/model.safetensors"]})


def arrays() -> list[str]:
    """Every array file under tests/fixtures, relative to it."""
    return sorted(str(path.relative_to(FIXTURES)) for path in FIXTURES.rglob("*")
                  if path.is_file() and path.name.endswith(ARRAYS))


def owners(path: str) -> list[str]:
    """The records that cover `path`: the reference naming it or its nearest
    directory, and input groups listing it."""
    found = [key for key in RECORDS if path == key or path.startswith(key + "/")]
    nearest = [max(found, key=len)] if found else []
    return nearest + [f"inputs from {group.get('tool') or group.get('from')}"
                      for group in INPUTS if path in group["files"]]


def named_in(source: str, name: str) -> bool:
    """Whether `source` names `name`: literally, or through a formatted
    string such as f"qwen38-{kind}-tiny" or f"video_{index}.npy"."""
    if name in source:
        return True
    templates = re.findall(r"""f["']([^"'\n]*\{[^"'\n]*)["']""", source)
    return any(re.fullmatch(re.sub(r"\\\{[^}]*\\\}", ".+", re.escape(template)), name)
               for template in templates)


@pytest.mark.parametrize("group", INPUTS, ids=lambda group: group.get("tool") or group.get("from"))
def test_an_input_group_names_what_wrote_its_files(group):
    """The group's tool exists and its source names the directory of each file
    it claims to write (a top-level file, by its own name)."""
    assert [path for path in group["files"] if not (FIXTURES / path).is_file()] == []
    if "from" in group:
        assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", group["from"]), group["from"]
        return
    assert (ROOT / group["tool"]).is_file(), group["tool"]
    source = (ROOT / group["tool"]).read_text()
    unnamed = [path for path in group["files"]
               if not named_in(source, Path(path).parent.name or Path(path).name.split(".")[0])]
    assert unnamed == []


def claimed() -> set[str]:
    """The directories a record claims whole."""
    return {key for key in RECORDS if (FIXTURES / key).is_dir()}


def writer(record: dict) -> str:
    return record.get("tool") or f"generator {record.get('generator')}"


def intruders(records: dict, directories: set[str]) -> dict[str, str]:
    """Each record inside a claimed directory that names another writer,
    with the directory. A claimed directory nested in another is its own."""
    found = {}
    for key, record in records.items():
        above = [directory for directory in directories if key.startswith(directory + "/")]
        if key in directories or not above:
            continue
        owner = max(above, key=len)
        if writer(record) != writer(records[owner]):
            found[key] = owner
    return found


def test_a_claimed_directory_holds_only_its_own_tools_files():
    """Two tools writing into one directory break the generator test that
    compares the directory's listing. Files the owning tool also records one
    by one pass; an input group's file inside a claimed directory already
    has two records and fails the test below. A file with no record at all
    counts as the owner's here: where tests/test_tools.py has a listing test
    for the directory, that test refuses it, since the tool does not write it."""
    assert intruders(RECORDS, claimed()) == {}
    moe = {"tool": "tools/moe_reference.py"}
    assert intruders({"moe": moe, "moe/solvers.npz": {"tool": "tools/flaxdiff_solver_reference.py"}},
                     {"moe"}) == {"moe/solvers.npz": "moe"}
    golden = {"generator": "dew"}
    assert intruders({"moe": moe, "moe/golden.npz": golden}, {"moe"}) == {"moe/golden.npz": "moe"}
    assert intruders({"moe": moe, "moe/tiny.npz": moe, "moe/sub": golden, "moe/sub/golden.npz": golden},
                     {"moe", "moe/sub"}) == {}


def test_every_array_file_has_exactly_one_record():
    """A file in no record is a fixture nobody has said the origin of; a file
    in two is either an input claimed as a reference or listed twice."""
    assert {path: owners(path) for path in arrays() if len(owners(path)) != 1} == {}
