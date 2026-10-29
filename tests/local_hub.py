"""A Hugging Face Hub on the loopback, for tests at the Hub's boundary without the network: it serves
repos of local files, each at one commit, and their open pull requests, each its own files at its own
commit on the repo's, over the HTTP routes `huggingface_hub` calls (a listing of its models, a repo's info
at a revision, its tree, a file's metadata and its bytes, a range of them as a safetensors header read
asks, a repo's discussions and a revision's commits), and keeps every request it answers, a missing repo's
or file's too (`LocalHub.served`), so a test counts what a load fetched. A process reaches it through
`HF_ENDPOINT` (`LocalHub.endpoint`), which `huggingface_hub` reads when it is imported."""

import hashlib
import json
import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit


@dataclass(frozen=True)
class Served:
    """One request the Hub answered: its method, the repo it named (None for the listing), the repo file
    (None for an API route), the byte range it asked (None for the whole file) and the bytes of the file
    it sent (none for a missing repo or file)."""

    method: str
    repo: str | None
    file: str | None
    asked: tuple[int, int] | None
    sent: int


@dataclass(frozen=True)
class Pull:
    """An open pull request on a repo: its title, its author's name and its files, which the Hub serves at
    its ref `refs/pr/<num>` and its own commit, whose parent is the repo's commit."""

    title: str
    author: str
    files: Mapping[str, Path]


class LocalHub:
    """`repos` (a repo id to its files, by repo-relative name) served on the loopback while the hub is
    entered, each repo at the commit its files hash to (`commit`), and `pulls` (a repo id to its open pull
    requests by number), each at its own commit (`pull_commit`)."""

    def __init__(self, repos: Mapping[str, Mapping[str, Path]],
                 pulls: Mapping[str, Mapping[int, Pull]] | None = None):
        self.repos = {repo: dict(files) for repo, files in repos.items()}
        self.pulls = {repo: dict(held) for repo, held in (pulls or {}).items()}
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
        return _hashed(self.repos[repo])

    def pull_commit(self, repo: str, num: int) -> str:
        """Pull request `num`'s commit on `repo`: a hash of its files and of the commit it sits on."""
        return _hashed(self.pulls[repo][num].files, self.commit(repo))

    def at(self, repo: str, revision: str | None) -> tuple[Mapping[str, Path], str]:
        """`repo`'s files and commit at `revision`: a pull request's at its ref or its commit, else the
        repo's own, whatever else the revision names."""
        for num in self.pulls.get(repo, {}):
            if revision in (f"refs/pr/{num}", self.pull_commit(repo, num)):
                return self.pulls[repo][num].files, self.pull_commit(repo, num)
        return self.repos[repo], self.commit(repo)

    def sent(self, repo: str, file: str) -> list[Served]:
        """The requests that sent bytes of `repo`'s `file` (a repo-relative name)."""
        return [served for served in self.served
                if (served.repo, served.file, served.method) == (repo, file, "GET")]


_LISTING = "/api/models"
_API = re.compile(r"/api/models/(?P<repo>[^/]+/[^/]+)(?:/revision/(?P<revision>[^/]+)"
                  r"|/tree/(?P<tree>[^/]+)(?:/(?P<folder>.+))?)?")
_FILE = re.compile(r"/(?P<repo>[^/]+/[^/]+)/resolve/(?P<revision>[^/]+)/(?P<file>.+)")
_DISCUSSIONS = re.compile(r"/api/models/(?P<repo>[^/]+/[^/]+)/discussions")
_COMMITS = re.compile(r"/api/models/(?P<repo>[^/]+/[^/]+)/commits/(?P<revision>[^/]+)")
_RANGE = re.compile(r"bytes=(\d+)-(\d*)")
_DATE = "2024-01-01T00:00:00.000Z"
"""The date every discussion and commit states."""


def _hashed(files: Mapping[str, Path], parent: str = "") -> str:
    """A commit's hash: its files' names and bytes, after its parent's."""
    digest = hashlib.sha1(parent.encode())
    for name, path in sorted(files.items()):
        digest.update(name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


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
            address = urlsplit(self.path)
            path = unquote(address.path)
            # A revision names a pull request's ref with its slashes quoted: the routes match the path as
            # sent, and each part is read unquoted.
            if self._pulls(address.path, head=head):
                return
            if path == _LISTING:
                # Every repo, whatever the listing filters, each with the fields `expand` asks, as the Hub
                # states a commit and downloads only where asked.
                hub.served.append(Served(self.command, None, None, None, 0))
                expand = parse_qs(address.query).get("expand", [])
                self._json([{"id": repo, "modelId": repo,
                             **({"sha": hub.commit(repo)} if "sha" in expand else {}),
                             **({"downloads": 0} if "downloads" in expand else {})} for repo in hub.repos],
                           head=head)
                return
            api, file = _API.fullmatch(address.path), _FILE.fullmatch(address.path)
            match = api or file
            if match is None or match["repo"] not in hub.repos:
                hub.served.append(Served(self.command, None if match is None else match["repo"],
                                         None if file is None else unquote(file["file"]), None, 0))
                self._missing("RepoNotFound", head=head)
                return
            repo = match["repo"]
            revision = (api["tree"] if api is not None and api["tree"] is not None else match["revision"])
            files, commit = hub.at(repo, None if revision is None else unquote(revision))
            if api is not None:
                hub.served.append(Served(self.command, repo, None, None, 0))
                folder = unquote(api["folder"] or "").rstrip("/")
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
            name = unquote(file["file"])
            if name not in files:
                hub.served.append(Served(self.command, repo, name, None, 0))
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

        def _pulls(self, sent: str, *, head: bool) -> bool:
            """Answer a repo's discussions (its open pull requests) or a revision's commits, the revision's
            and the one a pull request sits on; False for any other route."""
            discussions, commits = _DISCUSSIONS.fullmatch(sent), _COMMITS.fullmatch(sent)
            match = discussions or commits
            if match is None or match["repo"] not in hub.repos:
                return False
            repo = match["repo"]
            hub.served.append(Served(self.command, repo, None, None, 0))
            pulls = hub.pulls.get(repo, {})
            if discussions is not None:
                # Filtered by author where the request names one, as the Hub filters them.
                author = parse_qs(urlsplit(self.path).query).get("author")
                listed = [{"num": num, "title": pull.title, "status": "open", "author": {"name": pull.author},
                           "createdAt": _DATE, "repo": {"name": repo, "type": "model"}, "isPullRequest": True}
                          for num, pull in sorted(pulls.items()) if author is None or pull.author in author]
                self._json({"discussions": listed, "count": len(listed), "start": 0, "numItems": len(listed)},
                           head=head)
                return True
            _, commit = hub.at(repo, unquote(commits["revision"]))
            history = [commit] + ([hub.commit(repo)] if commit != hub.commit(repo) else [])
            self._json([{"id": sha, "title": "", "message": "", "authors": [], "date": _DATE}
                        for sha in history], head=head)
            return True

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
