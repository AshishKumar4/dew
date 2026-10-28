"""A Hugging Face Hub on the loopback, for tests at the Hub's boundary without the network: it serves
repos of local files, each at one commit, over the HTTP routes `huggingface_hub` calls (a repo's info,
its tree, a file's metadata and its bytes, a range of them as a safetensors header read asks), and keeps
every request it answers (`LocalHub.served`), so a test counts what a load fetched. A process reaches it
through `HF_ENDPOINT` (`LocalHub.endpoint`), which `huggingface_hub` reads when it is imported."""

import hashlib
import json
import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit


@dataclass(frozen=True)
class Served:
    """One request the Hub answered: its method, the repo it named, the repo file (None for an API
    route), the byte range it asked (None for the whole file) and the bytes of the file it sent."""

    method: str
    repo: str
    file: str | None
    asked: tuple[int, int] | None
    sent: int


class LocalHub:
    """`repos` (a repo id to its files, by repo-relative name) served on the loopback while the hub is
    entered, each repo at the commit its files hash to (`commit`)."""

    def __init__(self, repos: Mapping[str, Mapping[str, Path]]):
        self.repos = {repo: dict(files) for repo, files in repos.items()}
        self.served: list[Served] = []
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(self))
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> "LocalHub":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._server.shutdown()
        self._server.server_close()

    @property
    def endpoint(self) -> str:
        """The hub's base URL, what `HF_ENDPOINT` names."""
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def commit(self, repo: str) -> str:
        """`repo`'s commit: a hash of its files' names and bytes."""
        digest = hashlib.sha1()
        for name, path in sorted(self.repos[repo].items()):
            digest.update(name.encode() + b"\0" + path.read_bytes())
        return digest.hexdigest()

    def sent(self, repo: str, file: str) -> list[Served]:
        """The requests that sent bytes of `repo`'s `file` (a repo-relative name)."""
        return [served for served in self.served
                if (served.repo, served.file, served.method) == (repo, file, "GET")]


_API = re.compile(r"/api/models/(?P<repo>[^/]+/[^/]+)(?:/revision/(?P<revision>[^/]+)"
                  r"|/tree/(?P<tree>[^/]+)(?:/(?P<folder>.+))?)?")
_FILE = re.compile(r"/(?P<repo>[^/]+/[^/]+)/resolve/(?P<revision>[^/]+)/(?P<file>.+)")
_RANGE = re.compile(r"bytes=(\d+)-(\d*)")


def _handler(hub: LocalHub) -> type[BaseHTTPRequestHandler]:
    """The request handler of `hub`'s server: each route as the Hub answers it, unknown repos and files
    answered as the Hub answers them (404, with its error code)."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: str) -> None:
            """Quiet: the hub's record is `served`."""

        def do_HEAD(self) -> None:
            self._answer(head=True)

        def do_GET(self) -> None:
            self._answer(head=False)

        def _answer(self, *, head: bool) -> None:
            path = unquote(urlsplit(self.path).path)
            api, file = _API.fullmatch(path), _FILE.fullmatch(path)
            match = api or file
            if match is None or match["repo"] not in hub.repos:
                self._missing("RepoNotFound", head=head)
                return
            repo = match["repo"]
            files, commit = hub.repos[repo], hub.commit(repo)
            if api is not None:
                hub.served.append(Served(self.command, repo, None, None, 0))
                folder = (api["folder"] or "").rstrip("/")
                if api["tree"] is None:
                    self._json({"id": repo, "modelId": repo, "sha": commit, "private": False,
                                "siblings": [{"rfilename": name} for name in sorted(files)]}, head=head)
                elif folder and not any(name.startswith(f"{folder}/") for name in files):
                    self._missing("EntryNotFound", head=head, commit=commit)
                else:
                    self._json([{"type": "file", "path": name, "size": files[name].stat().st_size,
                                 "oid": hashlib.sha1(files[name].read_bytes()).hexdigest()}
                                for name in sorted(files) if name.startswith(f"{folder}/" if folder else "")],
                               head=head)
                return
            name = file["file"]
            if name not in files:
                self._missing("EntryNotFound", head=head, commit=commit)
                return
            body = files[name].read_bytes()
            asked = _RANGE.fullmatch(self.headers.get("range", "")) if not head else None
            start, stop = (0, len(body)) if asked is None else (
                int(asked[1]), min(len(body), int(asked[2]) + 1 if asked[2] else len(body)))
            sent = body[start:stop]
            hub.served.append(Served(self.command, repo, name, None if asked is None else (start, stop),
                                     0 if head else len(sent)))
            self.send_response(200 if asked is None else 206)
            self.send_header("X-Repo-Commit", commit)
            self.send_header("ETag", f'"{hashlib.sha1(body).hexdigest()}"')
            self.send_header("Content-Length", str(len(body) if head else len(sent)))
            if asked is not None:
                self.send_header("Content-Range", f"bytes {start}-{stop - 1}/{len(body)}")
            self.end_headers()
            if not head:
                self.wfile.write(sent)

        def _json(self, value, *, head: bool) -> None:
            body = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if not head:
                self.wfile.write(body)

        def _missing(self, code: str, *, head: bool, commit: str | None = None) -> None:
            body = json.dumps({"error": code}).encode()
            self.send_response(404)
            self.send_header("X-Error-Code", code)
            if commit is not None:
                self.send_header("X-Repo-Commit", commit)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if not head:
                self.wfile.write(body)

    return Handler
