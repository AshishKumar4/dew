"""A decoder Dew trained and exported, served by a live Ollama daemon.

Network-marked, so an ordinary `-m "not network"` run deselects it. Nothing
downloads: the model is one this file trains through the LM recipe, and the
tokenizer is the committed byte-level BPE fixture the token files were
written with.

What is held to account is the artifact contract. `ollama create` reads the
HF layout `save_pretrained_decoder` writes, so the GGUF its converter emits
carries the exported config's own numbers and serves the computation the
export describes: asked for the model's own argmax, the daemon reproduces
Dew's greedy draw token for token. That is what catches a conversion which
moved the computation rather than repacking it; a transposed kernel or the
other rope convention answers with different tokens from the first step.

Asking for the model's own argmax means naming `repeat_penalty`. The daemon
applies 1.1 over the last tokens by default, and the client passes backend
options through untouched, so the raw policy is reachable but never the
default. One test pins the difference between the two.

The daemon exposes no tokenize endpoint, so what a test here can compare is
`prompt_eval_count` against the HF tokenizer's length, not the ids. Length
agreement is not id agreement, and the two tokenizers genuinely part ways:
the GGUF records the pre-tokenizer as `default`, whose regex cuts digit runs
into groups of three, where the exported ByteLevel tokenizer keeps a run
whole. One test pins that too, so nobody reads the length check as identity.

`save_pretrained_decoder` is the low-level writer and writes no tokenizer
assets; it records the name under `tokenizer_name` in generation_config.json
and nothing else. `Pretrained.save` is the path that also writes the
attached processor (`pretrained.py:428-429`), so an export routed through it
needs no copying. This file exercises the low-level writer, hence the copy,
and pins the bare directory being refused.

The model is created under a name generated for the run and removed when
the module finishes; no other model the daemon holds is touched.
`num_gpu 0` keeps the inference on the CPU, so no test here needs a device
the training run did not already use.
"""

import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from urllib import error, request

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.data import HFTokenizer
from dew.inference import Completion, OllamaCompletion
from dew.interop import load_pretrained
from dew.interop.hf_decoders import save_pretrained_decoder
from dew.nn.inputs import ModelInputs
from dew.registry import models, with_precision
from dew.sampling.text import Sampling, generate
from test_lm_recipe import load_recipe

pytestmark = pytest.mark.network

REPO_ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = REPO_ROOT / "tests" / "fixtures" / "tokenizers" / "tiny-tools"
CORPUS = ("README.md", "CONTRIBUTING.md", "docs/research/inference.md")
SEQ = 128
STEPS = 400
FIELDS = {"emb_features": 128, "num_layers": 4, "num_heads": 4, "num_kv_heads": 2,
          "mlp_features": 256, "max_seq_len": SEQ, "qk_norm": False,
          "tie_embeddings": False}
PROMPTS = ("The trainer", "Dew trains", "The model", "This recipe")
DRAWN = 8


def daemon_url() -> str:
    """Where the daemon listens, under the variable its own client reads."""
    host = os.environ.get("OLLAMA_HOST") or "127.0.0.1:11434"
    return host if host.startswith("http") else f"http://{host}"


def ask(path: str, payload: dict | None = None, timeout: float = 300.0) -> dict:
    body = None if payload is None else json.dumps(payload).encode()
    call = request.Request(f"{daemon_url()}{path}", data=body,
                           headers={"Content-Type": "application/json"},
                           method="GET" if body is None else "POST")
    with request.urlopen(call, timeout=timeout) as response:
        return json.loads(response.read())


def draw(client: OllamaCompletion, prompt: str, count: int,
         options: dict[str, object] | None = None, **fields: object) -> Completion:
    """One greedy raw completion through the client under test.

    The two namespaces stay apart on purpose. `options` carries backend
    sampler options, where the client adds `num_predict` and `seed` and
    refuses a caller's own; `fields` carries SDK request fields such as `raw`
    and `logprobs`. A request field misplaced into `options` is dropped in
    silence, since the SDK's Options model ignores what it does not declare.
    """
    return client(prompt, count, seed=0, raw=True,
                  options={"temperature": 0, **(options or {})}, **fields)


def prompt_tokens(client: OllamaCompletion, prompt: str) -> int | None:
    """How many tokens the daemon charged the prompt, off the retained SDK
    response. The count is all it reports; the daemon exposes no tokenize
    endpoint, so the ids it used stay hidden."""
    return draw(client, prompt, 1).responses[0].prompt_eval_count


def write_token_files(directory: Path, tokenizer: HFTokenizer) -> dict:
    """The train.bin / val.bin / meta.json a tokenizer run writes, over the
    repo's own prose. Real text, so the trained model prefers a definite next
    token and the greedy comparison below is not a coin flip."""
    text = "\n".join((REPO_ROOT / name).read_text() for name in CORPUS)
    ids = np.asarray(tokenizer.encode(text), np.uint16)
    assert ids.size > 64 * SEQ, f"{ids.size} tokens is too little to train on"
    split = ids.size // 20
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "val.bin").write_bytes(ids[:split].tobytes())
    (directory / "train.bin").write_bytes(ids[split:].tobytes())
    meta = {"tokenizer": str(TOKENIZER), "vocab_size": tokenizer.vocab_size,
            "dtype": "uint16", "eos_id": None}
    (directory / "meta.json").write_text(json.dumps(meta))
    return meta


def train_and_export(root: Path) -> Path:
    """A short real run on those tokens, written back out in the HF layout
    with the tokenizer that produced the ids copied beside it."""
    import tyro

    recipe = load_recipe()
    tokenizer = HFTokenizer(str(TOKENIZER))
    tokens = root / "tokens"
    meta = write_token_files(tokens, tokenizer)
    config = tyro.cli(tyro.conf.CascadeSubcommandArgs[recipe.LmRunConfig], args=[
        "--data.path", str(tokens), "--data.seq-len", str(SEQ),
        "--data.loading.workers", "0", "--tokenizer", str(TOKENIZER),
        "--trainer.batch-size", "16", "--trainer.steps", str(STEPS),
        "--trainer.log-every", str(STEPS), "--trainer.eval-every", "None",
        "--trainer.checkpoint-every", "None", "--trainer.name", "ollama-interop",
        "--trainer.checkpoint-dir", str(root / "runs"),
        "--trainer.compilation-cache-dir", "None", "--trainer.multi-host", "False",
        "--model.dtype", "float32", "--sample-tokens", "0", "--ema-decay", "None",
        "--optim.learning-rate", "3e-3", "--optim.learning-rate-warmup-steps", "50",
        "--model.config", json.dumps(FIELDS)])
    state = recipe.main(config)
    assert int(state.step) == STEPS

    model = models.build(config.model.architecture, **with_precision(
        config.model.architecture, {**FIELDS, "vocab_size": meta["vocab_size"]},
        dtype="float32", attention_impl="xla"))
    export = root / "export"
    save_pretrained_decoder(model, state.params, str(export),
                            tokenizer_name=str(TOKENIZER))
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy2(TOKENIZER / name, export / name)
    return export


def create(name: str, export: Path, root: Path) -> subprocess.CompletedProcess:
    """`ollama create` over a Modelfile, the supported conversion path for a
    Safetensors directory. num_gpu pins the runner to the CPU."""
    modelfile = root / f"Modelfile.{name}"
    modelfile.write_text(f"FROM {export}\nTEMPLATE \"{{{{ .Prompt }}}}\"\n"
                         f"PARAMETER num_gpu 0\nPARAMETER num_ctx {SEQ}\n")
    return subprocess.run(["ollama", "create", name, "-f", str(modelfile)],
                          capture_output=True, text=True, timeout=900, check=False)


def remove(name: str) -> None:
    """Drop one model by name, whatever state a failed create left it in.
    Nothing else the daemon holds is a test's to delete."""
    subprocess.run(["ollama", "rm", name], capture_output=True, text=True,
                   timeout=300, check=False)


@pytest.fixture(scope="module")
def imported(tmp_path_factory):
    """The trained export, converted into a model of this run's own name.

    Yields the name, the export directory and the daemon's view of the
    converted model. The name is removed on teardown, and only that name.
    """
    if shutil.which("ollama") is None:
        pytest.skip("the ollama binary is not on PATH")
    try:
        version = ask("/api/version", timeout=10.0)["version"]
    except (error.URLError, OSError) as failure:
        pytest.skip(f"no ollama daemon at {daemon_url()}: {failure}")

    root = tmp_path_factory.mktemp("ollama-interop")
    export = train_and_export(root)
    name = f"dew-interop-{uuid.uuid4().hex[:8]}"
    try:
        done = create(name, export, root)
        assert done.returncode == 0, (
            f"ollama {version} refused the export:\n{done.stderr}")
        yield name, export, ask("/api/show", {"model": name})
    finally:
        remove(name)


@pytest.fixture(scope="module")
def client(imported):
    """`OllamaCompletion` bound to the imported model and an injected SDK
    client. The SDK is the `inference-clients` extra, so a checkout without
    it skips rather than fails."""
    ollama = pytest.importorskip("ollama", reason="pip install dew-ml[inference-clients]")
    name, _, _ = imported
    return OllamaCompletion(name, ollama.Client(host=daemon_url()))


def test_the_converted_model_carries_the_exported_config(imported):
    """Ollama's converter reads the export's own config, so the GGUF it
    writes reports Dew's widths, its rope base and its parameter count.
    A field the export spelled wrong lands here as a different number."""
    name, export, shown = imported
    config = json.loads((export / "config.json").read_text())
    info = shown["model_info"]

    assert info["general.architecture"] == "llama"
    assert info["llama.embedding_length"] == config["hidden_size"]
    assert info["llama.block_count"] == config["num_hidden_layers"]
    assert info["llama.attention.head_count"] == config["num_attention_heads"]
    assert info["llama.attention.head_count_kv"] == config["num_key_value_heads"]
    assert info["llama.attention.head_dim"] == config["head_dim"]
    assert info["llama.feed_forward_length"] == config["intermediate_size"]
    assert info["llama.context_length"] == config["max_position_embeddings"]
    assert info["llama.rope.freq_base"] == config["rope_theta"]
    assert info["llama.attention.layer_norm_rms_epsilon"] == config["rms_norm_eps"]
    assert info["llama.vocab_size"] == config["vocab_size"]

    weights = sum(int(np.prod(entry["shape"])) for entry in shown["tensors"])
    assert info["general.parameter_count"] == weights


def test_the_tokenizer_reaches_the_gguf(imported, client):
    """The tokenizer copied beside the export is the one the daemon loads:
    its end and unknown ids are in the GGUF, and a prompt costs the daemon
    the number of tokens the HF tokenizer encodes it into.

    A count, not the ids. The daemon serves no tokenize endpoint, so the ids
    it used are not observable here; a tokenizer that failed to convert
    would still show up, since its lengths would not line up.
    """
    _, export, shown = imported
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(export), local_files_only=True)
    info = shown["model_info"]
    assert info["tokenizer.ggml.model"] == "gpt2", "byte-level BPE reads as gpt2"
    assert info["tokenizer.ggml.eos_token_id"] == tokenizer.eos_token_id
    assert info["tokenizer.ggml.unknown_token_id"] == tokenizer.unk_token_id

    for prompt in PROMPTS:
        assert prompt_tokens(client, prompt) == len(
            tokenizer.encode(prompt, add_special_tokens=False)), prompt


def test_a_digit_run_tokenizes_differently_in_the_gguf(imported, client):
    """Why the length check above is not an identity claim.

    The converter records the pre-tokenizer as `default`, whose regex cuts a
    digit run into groups of three; the exported ByteLevel tokenizer keeps
    the run whole and merges across it. So `0123` is three tokens to Dew and
    four to the daemon, and ids Dew wrote are not the ids the served model
    reads. A converter that learned to carry the ByteLevel regex would make
    these equal, and this test is how that would be noticed.
    """
    _, export, _ = imported
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(export), local_files_only=True)
    ours = tokenizer.encode("0123", add_special_tokens=False)

    assert len(ours) == 3, tokenizer.convert_ids_to_tokens(ours)
    assert prompt_tokens(client, "0123") == 4


def test_the_client_completes_prompts_against_the_daemon(client):
    """`OllamaCompletion` over the imported model: one answer per prompt in
    prompt order, the token budget spent, the backend's own finish reason,
    and the SDK responses retained. Ollama reports no aggregate usage."""
    answer = client(list(PROMPTS), DRAWN, seed=1234,
                    options={"temperature": 0.8, "top_k": 40})

    assert len(answer.texts) == len(PROMPTS)
    assert all(text for text in answer.texts)
    assert answer.token_counts == (DRAWN,) * len(PROMPTS)
    assert answer.finish_reasons == ("length",) * len(PROMPTS)
    assert answer.usage is None
    assert len(answer.responses) == len(PROMPTS)


def test_zero_temperature_repeats_itself(client):
    """The seed rides on every call, so the same greedy request twice is the
    same text: a completion is reproducible against a fixed model."""
    first = client(PROMPTS[0], DRAWN, seed=7, options={"temperature": 0.0})
    again = client(PROMPTS[0], DRAWN, seed=7, options={"temperature": 0.0})

    assert first.texts == again.texts


def test_the_daemon_draws_dews_own_greedy_continuation(imported, client):
    """The conversion repacked the computation, it did not change it.

    The export is read back through `load_pretrained`, drawn from at
    temperature zero, and compared with the daemon asked for the same thing:
    argmax over the model's own policy, so `repeat_penalty` is set to 1. The
    two agree token for token. Ollama serves F16 tensors through llama.cpp's
    kernels and Dew scores fp32 through XLA, and that difference is far too
    small to move an argmax here; a transposed kernel or the other rope
    convention answers with a different token immediately.
    """
    _, export, _ = imported
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(export), local_files_only=True)
    loaded = load_pretrained(str(export), dtype="float32", attention_impl="xla")

    for prompt in PROMPTS:
        head = tokenizer.encode(prompt, add_special_tokens=False)
        drawn = generate(loaded.model, loaded.variables,
                         ModelInputs(tokens=jnp.asarray([head], jnp.int32)), DRAWN,
                         key=jax.random.key(0), sampling=Sampling(temperature=0.0))
        ours = [int(token) for token in np.asarray(drawn.tokens)[0, len(head):]]
        answer = draw(client, prompt, DRAWN, {"repeat_penalty": 1.0})
        theirs = tokenizer.encode(answer.texts[0], add_special_tokens=False)

        assert theirs == ours, (f"{prompt!r}: daemon {theirs} "
                                f"({answer.texts[0]!r}) against dew {ours} "
                                f"({tokenizer.decode(ours)!r})")


def test_the_daemon_reports_dews_own_logprobs(imported, client):
    """Tighter than the token identity above, and it drifts first.

    `/api/generate` answers `logprobs` with the log-probability it assigned
    each token it drew, so the two engines can be compared as numbers rather
    than as decisions. F16 tensors under llama.cpp's kernels against fp32
    under XLA is worth a few thousandths of a logprob here, far less than
    what separates the model's first choice from its second, which is why
    the argmax agreement above is not luck. A kernel or precision change
    that started to matter would widen this before it flipped a token.
    """
    _, export, _ = imported
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(export), local_files_only=True)
    loaded = load_pretrained(str(export), dtype="float32", attention_impl="xla")

    worst = 0.0
    for prompt in PROMPTS:
        head = tokenizer.encode(prompt, add_special_tokens=False)
        answer = draw(client, prompt, DRAWN, {"repeat_penalty": 1.0}, logprobs=True)
        reported = answer.responses[0].logprobs
        assert reported, f"the daemon reported no logprobs for {prompt!r}"
        theirs = tokenizer.encode(answer.texts[0], add_special_tokens=False)
        assert len(theirs) == len(reported), prompt

        logits = np.asarray(loaded.model.apply(
            loaded.variables, jnp.asarray([head + theirs], jnp.int32)))[0]
        for offset, (token, entry) in enumerate(zip(theirs, reported)):
            row = logits[len(head) + offset - 1].astype(np.float64)
            ours = float(row[token] - (np.log(np.exp(row - row.max()).sum()) + row.max()))
            worst = max(worst, abs(ours - float(entry.logprob)))

    assert worst < 0.05, f"max |dew - daemon| logprob {worst:.5f}"


def test_the_daemons_default_penalty_is_not_the_models_policy(client):
    """What a caller who names no penalty gets, and why the test above names
    one.

    Temperature zero is argmax over a distribution the daemon has already
    penalised: `repeat_penalty` defaults to 1.1 over the last tokens. The
    client passes backend options through, so the raw policy is reachable,
    but only by asking; on this checkpoint the two answers differ.
    """
    penalised = draw(client, PROMPTS[0], DRAWN).texts
    raw = draw(client, PROMPTS[0], DRAWN, {"repeat_penalty": 1.0}).texts

    assert penalised != raw
    assert draw(client, PROMPTS[0], DRAWN, {"repeat_penalty": 1.1}).texts == penalised


def test_ollama_refuses_an_export_without_tokenizer_files(imported, tmp_path):
    """The gap `save_pretrained_decoder` leaves.

    It writes config.json, model.safetensors and a generation_config that
    names a tokenizer it does not copy. That directory alone does not
    convert: the tokenizer has to be put there by whoever exports.
    """
    _, export, _ = imported
    bare = tmp_path / "bare"
    bare.mkdir()
    for name in ("config.json", "model.safetensors", "generation_config.json"):
        shutil.copy2(export / name, bare / name)
    assert not (bare / "tokenizer.json").exists()

    probe = f"dew-interop-bare-{uuid.uuid4().hex[:8]}"
    try:
        done = create(probe, bare, tmp_path)
    finally:
        remove(probe)

    assert done.returncode != 0
    assert "tokenizer" in (done.stderr + done.stdout).lower()
