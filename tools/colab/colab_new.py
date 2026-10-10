"""Allocate idempotently and release only a journaled assignment.

The SDK is loaded only for live operations. Credentials stay in the CLI's
private store; no connection data enters a batch plan or result receipt.
"""

import importlib
import time
import uuid
from pathlib import Path

import requests

from tools.colab.common import STATE, lock, name, read, save


class Session:
    def __init__(self, endpoint: str, url: str, token: str):
        self.endpoint = endpoint
        self.url = url
        self.headers = {"X-Colab-Client-Agent": "colab-cli", "X-Colab-Runtime-Proxy-Token": token}
        self.token = token


class Colab:
    def __init__(self):
        self.state = importlib.import_module("colab_cli.common").state
        self.types = importlib.import_module("colab_cli.client")
        self.session_type = importlib.import_module("colab_cli.state").SessionState

    def assignments(self) -> list:
        return self.state.client.list_assignments()

    def session(self, endpoint: str) -> Session:
        assigned = next((item for item in self.assignments() if item.endpoint == endpoint), None)
        if assigned is None:
            raise LookupError("owned assignment absent")
        info = assigned.runtime_proxy_info
        return Session(endpoint, info.url, info.token)

    def allocate(self, session_name: str, hardware: str, high_mem: bool, journal: Path) -> str:
        session_name = name(session_name)
        with lock(STATE / "colab-allocation.lock"):
            held = self.assignments()
            named = self.state.store.get(session_name)
            if len(held) >= 4 or (named and any(item.endpoint == named.endpoint for item in held)):
                raise RuntimeError("no slot or named assignment already owned")
            resolve = importlib.import_module("colab_cli.commands.session").resolve_runtime_options
            variant, accelerator, shape = resolve(hardware.removeprefix("gpu:"), None, high_mem=high_mem)
            request = {
                "name": name(session_name),
                "notebook": str(uuid.uuid4()),
                "variant": variant.name,
                "accelerator": accelerator.name,
                "shape": shape.name if shape else None,
                "requested_at": time.time(),
            }
            save(journal, request)
            response = self._assign(request)
            self._store(session_name, response, variant, accelerator, shape)
            request.update(endpoint=response.endpoint, assigned_at=time.time())
            save(journal, request)
            return response.endpoint

    def _assign(self, request: dict):
        options = self._options(request)
        for attempt in range(12):
            try:
                return self.state.client.assign(uuid.UUID(request["notebook"]), **options)
            except requests.exceptions.ReadTimeout:
                if attempt == 11:
                    raise
                time.sleep(5)
        raise AssertionError("allocation retries exhausted")

    def _options(self, request: dict) -> dict:
        return {
            "variant": self.types.Variant[request["variant"]],
            "accelerator": self.types.Accelerator[request["accelerator"]],
            "shape": self.types.Shape[request["shape"]] if request["shape"] else None,
        }

    def recover_endpoint(self, journal: Path) -> str | None:
        request = read(journal)
        if not request:
            return None
        if request.get("endpoint"):
            return request["endpoint"]
        assigned = self.state.client._get_assignment(uuid.UUID(request["notebook"]), **self._options(request))
        return getattr(assigned, "endpoint", None)

    def _store(self, session_name, response, variant, accelerator, shape):
        info = response.runtime_proxy_info
        self.state.store.add(
            self.session_type(
                name=session_name,
                endpoint=response.endpoint,
                token=info.token,
                url=info.url,
                token_expires_at=info.expires_at(),
                variant=variant.value,
                accelerator=accelerator.value,
                machine_shape=(shape or self.types.Shape.STANDARD).name,
            )
        )

    def release(self, session_name: str, endpoint: str | None) -> str:
        if not endpoint:
            return "no assignment acquired"
        present = any(item.endpoint == endpoint for item in self.assignments())
        if present:
            self.state.client.unassign(endpoint)
        local = self.state.store.get(session_name)
        if local and local.endpoint == endpoint:
            self.state.store.remove(session_name)
        return "released" if present else "already absent"

    def kernel(self, session: Session, **kwargs):
        runtime = importlib.import_module("colab_cli.runtime").ColabRuntime
        return runtime(session.url, session.token, **kwargs)
