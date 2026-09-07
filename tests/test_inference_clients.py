"""External completion clients speak the engines' wire formats over a real socket."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from dew.inference import Completion, OllamaCompletion, OpenAICompletion
from dew.sampling import Sampling


class _Server(HTTPServer):
    """Records requests and answers with a scripted body per path."""

    def __init__(self, scripts):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.scripts = scripts
        self.requests = []
        self.thread = threading.Thread(target=self.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_address[1]}"

    def close(self):
        self.shutdown()
        self.server_close()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_POST(self):
        server = self.server
        assert isinstance(server, _Server)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        server.requests.append((self.path, dict(self.headers), body))
        status, answer = server.scripts[self.path](body)
        payload = json.dumps(answer).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def server():
    servers = []

    def start(scripts):
        started = _Server(scripts)
        servers.append(started)
        return started

    yield start
    for started in servers:
        started.close()


def test_ollama_client_sends_its_options_and_reads_each_answer(server):
    def generate(body):
        assert body["stream"] is False
        seed = body["options"]["seed"]
        return 200, {"response": f"{body['prompt']}|{seed}", "done_reason": "stop" if seed % 2 else "length",
                     "eval_count": seed * 10}

    started = server({"/api/generate": generate})
    client = OllamaCompletion("tiny", url=started.url)
    completion = client(["a", "b"], 12, seed=41, sampling=Sampling(temperature=0.3, top_k=7))
    assert completion == Completion(("a|41", "b|42"), (True, False), (410, 420))
    assert [body["model"] for _, _, body in started.requests] == ["tiny", "tiny"]
    assert started.requests[0][2]["options"] == {"temperature": 0.3, "num_predict": 12, "seed": 41, "top_k": 7}
    single = client("only", 3, seed=1)
    assert single.texts == ("only|1",) and "top_k" not in started.requests[-1][2]["options"]


def test_openai_client_batches_prompts_and_orders_choices(server):
    def completions(body):
        assert body["prompt"] == ["x", "y"] and body["max_tokens"] == 5 and body["seed"] == 9
        return 200, {"choices": [
            {"index": 1, "text": "second", "finish_reason": "length"},
            {"index": 0, "text": "first", "finish_reason": "stop"}],
            "usage": {"completion_tokens": 7}}

    started = server({"/v1/completions": completions})
    client = OpenAICompletion("served", url=started.url, api_key="secret")
    completion = client(["x", "y"], 5, seed=9, sampling=Sampling(temperature=0.5, top_k=3))
    assert completion.texts == ("first", "second") and completion.finished == (True, False)
    assert completion.token_counts == (0, 0)
    path, headers, body = started.requests[0]
    assert path == "/v1/completions" and headers["Authorization"] == "Bearer secret"
    assert body["temperature"] == 0.5 and body["top_k"] == 3 and body["model"] == "served"


def test_controls_the_engine_cannot_honor_are_refused_before_any_request(server):
    started = server({"/api/generate": lambda body: (200, {"response": ""})})
    client = OllamaCompletion("tiny", url=started.url)
    with pytest.raises(ValueError, match="eos_id"):
        client("a", 4, seed=0, sampling=Sampling(eos_id=2))
    with pytest.raises(ValueError, match="pad id"):
        client("a", 4, seed=0, sampling=Sampling(pad_id=1))
    assert started.requests == []


def test_server_errors_and_malformed_answers_surface(server):
    started = server({
        "/api/generate": lambda body: (500, {"error": "model not found"}),
        "/v1/completions": lambda body: (200, {"choices": [{"index": 0, "text": "one"}]}),
    })
    with pytest.raises(RuntimeError, match="500"):
        OllamaCompletion("missing", url=started.url)("a", 4, seed=0)
    with pytest.raises(RuntimeError, match="1 choices for 2 prompts"):
        OpenAICompletion("served", url=started.url)(["a", "b"], 4, seed=0)
