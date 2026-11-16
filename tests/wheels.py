"""Wheels for tests that install with a real installer and no network: one built in place (`wheel`), and the
PEP 517 build hooks (`build_wheel`, `build_editable`) of a stand-in checkout, which copies this file in as its
own build backend."""

import base64
import hashlib
import zipfile
from pathlib import Path


def distribution(name: str, files: dict[str, str], version: str = "0.1") -> dict[str, str]:
    """The distribution `name` at `version` holding `files` (each path to its text), with its metadata and
    the record an installer writes of every file's digest: each path to its text, as a wheel holds them
    (`wheel`) and a site holds them installed (`installed`)."""
    info = f"{name.replace('-', '_')}-{version}.dist-info"
    files = {**files, f"{info}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
             f"{info}/WHEEL": ("Wheel-Version: 1.0\nGenerator: tests\nRoot-Is-Purelib: true\n"
                               "Tag: py3-none-any\n")}

    def digest(text: str) -> str:
        return base64.urlsafe_b64encode(hashlib.sha256(text.encode()).digest()).rstrip(b"=").decode()
    record = "".join(f"{file},sha256={digest(text)},{len(text.encode())}\n" for file, text in files.items())
    return {**files, f"{info}/RECORD": f"{record}{info}/RECORD,,\n"}


def installed(site: Path, name: str, files: dict[str, str], version: str = "0.1") -> None:
    """The distribution `name` holding `files` (`distribution`) written into `site` as an installer leaves
    it."""
    for file, text in distribution(name, files, version).items():
        (Path(site) / file).parent.mkdir(parents=True, exist_ok=True)
        (Path(site) / file).write_text(text)


def wheel(directory: Path, name: str, version: str = "0.1") -> Path:
    """A wheel of the package `name` at `version` written into `directory`: its one module, `__version__`
    naming the version, and its metadata and record, as an installer reads them (`distribution`)."""
    module = name.replace("-", "_")
    path = Path(directory) / f"{module}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for file, text in distribution(name, {f"{module}/__init__.py": f"__version__ = {version!r}\n"},
                                       version).items():
            archive.writestr(file, text)
    return path


def build_wheel(directory, config_settings=None, metadata_directory=None) -> str:
    """The stand-in checkout's own wheel, the package `checkout`: a PEP 517 hook its pyproject names."""
    return wheel(Path(directory), "checkout").name


build_editable = build_wheel
