"""Administrative Kernel Gateway startup/memory measurements on shared compute."""

import argparse
import concurrent.futures
import json
import pathlib
import statistics
import time
import urllib.request

from jupyter_client import BlockingKernelClient
from jupyter_core.utils import ensure_event_loop


def dump_processes():
    for proc in pathlib.Path('/proc').glob('[0-9]*'):
        try:
            command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            text = (proc / 'status').read_text() + '\ncmd=' + command
            if '6100' in text or '6101' in text:
                print('PROCESS_STATUS',text[:1600],flush=True)
        except OSError:
            pass

def request(path, data=None, method=None):
    token = pathlib.Path("/run/dew/gateway-token").read_text()
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request("http://127.0.0.1:8890" + path, data=body, method=method,
                                 headers={"Authorization": "token " + token,
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=40) as response:
        raw = response.read()
        return json.loads(raw) if raw else None


def execute(client, code):
    return collect(client, client.execute(code, allow_stdin=False))


def collect(client, message):
    stdout = ""
    while True:
        result = client.get_iopub_msg(timeout=30)
        if result["parent_header"].get("msg_id") != message:
            continue
        if result["msg_type"] == "stream":
            stdout += result["content"]["text"]
        if result["msg_type"] == "error":
            raise RuntimeError(str(result["content"]))
        if result["msg_type"] == "status" and result["content"]["execution_state"] == "idle":
            return stdout


TRAINING_PROBE = """
import live_training
live_training.install()
for score in ("-1000", "-999", "0"):
    try:
        with open("/proc/self/oom_score_adj", "w") as file:
            file.write(score)
    except PermissionError:
        pass
    else:
        raise AssertionError(f"a guest lowered its OOM score to {score}")
assert open("/proc/self/oom_score_adj").read().strip() == "1000"
import urllib.request
from pathlib import Path
try:
    urllib.request.urlopen("https://huggingface.co", timeout=5)
except OSError:
    pass
else:
    raise AssertionError("a training context reached the network")
for path in ("/opt/train/escape", "/opt/live/escape", "/work/../escape"):
    try:
        Path(path).write_text("x")
    except OSError:
        pass
    else:
        raise AssertionError(f"a training context wrote {path}")
Path("/tmp/inside").write_text("x")
try:
    bytearray(7 * 1024 * 1024 * 1024)
except MemoryError:
    pass
else:
    raise AssertionError("a training context exceeded its memory allowance")
from dew import Trainer
from dew.config import OptimConfig
from dew.data import load
from dew.nn.backbones import CausalTransformer
from dew.objectives.lm import LMObjective
data = load("hf/winglian/tiny-shakespeare", batch=64, tokenizer="byte", seq_len=32)
model = CausalTransformer(vocab_size=256, emb_features=32, num_layers=1, num_heads=2,
                          mlp_features=64, max_seq_len=64)
state = Trainer(LMObjective(model, seq_len=32), OptimConfig(learning_rate=1e-3), key=0).fit(data, steps=1000)
assert int(state.step) == 20
from dew.interop import PretrainedDecoder
from dew.sampling import Sampling
model = PretrainedDecoder.load("HuggingFaceTB/SmolLM2-135M-Instruct", dtype="float32", max_seq_len=256)
assert model.text_generation(sampling=Sampling(temperature=0))("ROMEO:", 4, key=0).text[0]
"""


def training():
    """A training context reaches neither the network nor any path outside its scratch, reads the
    prepared corpus offline, and says first what its caps changed."""
    kernel = request("/api/kernels", {"name": "dew-train"})
    client = BlockingKernelClient(connection_file=f"/run/dew/gateway/kernel-{kernel['id']}.json")
    client.load_connection_file()
    client.start_channels()
    try:
        client.wait_for_ready(timeout=30)
        lines = execute(client, TRAINING_PROBE).splitlines()
        assert lines[0] == "Live run: batch 64 -> 8, so it fits a shared 4-vCPU host.", lines
        assert lines[1] == ("Live run: steps 1000 -> 20, log_every 100 -> 5, "
                            "so it finishes in about two minutes."), lines
    finally:
        client.stop_channels()
        request("/api/kernels/" + kernel["id"], method="DELETE")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("count", type=int, choices=(1, 10, 50))
    args = parser.parse_args()
    kernels = []
    try:
        for _ in range(min(args.count, 8)):
            start = time.perf_counter()
            model = request("/api/kernels", {"name": "python3"})
            connection = pathlib.Path("/run/dew/gateway") / f"kernel-{model['id']}.json"
            client = BlockingKernelClient(connection_file=str(connection))
            client.load_connection_file()
            client.start_channels()
            kernels.append((model["id"], client, 0))
            try:
                client.wait_for_ready(timeout=30)
            except Exception:
                print('KERNEL_STATUS',request('/api/kernels/'+model['id']),flush=True)
                dump_processes()
                raise
            kernels[-1]=(model["id"],client,time.perf_counter()-start)
            execute(client, f"from pathlib import Path\n"
                    f"assert '__measurement' not in globals()\n"
                    f"assert not Path('/work/context').exists()\n"
                    f"__measurement = {model['id']!r}\n"
                    f"Path('/work/context').write_text(__measurement)")
        for position, (identifier, client, _) in enumerate(kernels):
            foreign = kernels[(position + 1) % len(kernels)][0]
            control = client.session.msg('kernel_info_request', {})
            client.control_channel.send(control)
            reply = client.get_control_msg(timeout=30)
            assert reply['msg_type'] == 'kernel_info_reply'
            assert reply['parent_header']['msg_id'] == control['header']['msg_id']
            message = client.execute("print(input('probe input:'))", allow_stdin=True)
            assert client.get_stdin_msg(timeout=30)['msg_type'] == 'input_request'
            client.input('private stdin probe')
            assert collect(client, message).strip() == 'private stdin probe'
            execute(client,
                "import errno, os, resource, socket\n"
                "for proc in Path('/proc').glob('[0-9]*'):\n"
                " try:\n  uid = next(line for line in (proc / 'status').read_text().splitlines() "
                "if line.startswith('Uid:')).split()[1]\n"
                " except (OSError, StopIteration):\n  continue\n"
                " assert int(uid) == os.getuid(), 'guest saw a foreign process'\n"
                "links = {line.split(':')[0].strip() for line in "
                "Path('/proc/net/dev').read_text().splitlines()[2:]}\nassert links == {'lo'}\n"

                "status = Path('/proc/self/status').read_text().splitlines()\n"
                "assert all(line.split(':')[1].strip() == '0000000000000000' "
                "for line in status if line.startswith('Cap'))\n"
                "assert next(line for line in status if line.startswith('NoNewPrivs:')).split()[1] == '1'\n"
                "try:\n socket.create_connection(('127.0.0.1', 8890), timeout=1)\n"
                "except OSError:\n pass\nelse:\n raise AssertionError('guest reached gateway TCP')\n"
                "try:\n bytearray(1024 * 1024 * 1024)\n"
                "except MemoryError:\n pass\nelse:\n"
                " raise AssertionError('guest exceeded memory allowance')\n"
                "try:\n resource.setrlimit(resource.RLIMIT_AS, "
                "(resource.RLIM_INFINITY, resource.RLIM_INFINITY))\n"
                "except (PermissionError, ValueError):\n pass\n"
                "else:\n raise AssertionError('guest raised hard memory limit')\n"
                "try:\n pid = os.fork()\n"
                "except PermissionError:\n pass\nelse:\n"
                " if pid == 0: os._exit(0)\n os.waitpid(pid, 0)\n raise AssertionError('guest forked')")
            if foreign != identifier:
                execute(client,
                    f"for path in ['/sessions/connections/kernel-{foreign}.json', "
                    f"'/sessions/ipc/kernel-{foreign}/kernel-1']:\n"
                    " try:\n  Path(path).read_bytes()\n"
                    " except PermissionError:\n  pass\n"
                    " else:\n  raise AssertionError('guest read another context')\n"
                    "sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
                    f"try:\n sock.connect('/sessions/ipc/kernel-{foreign}/kernel-1')\n"
                    "except PermissionError:\n pass\n"
                    "else:\n raise AssertionError('guest connected to another context')\n"
                    "finally:\n sock.close()")
        for _, client, _ in kernels:
            execute(client,
                "import model_client\nmodel_client.install()\n"
                "from dew.sampling import CFG, DPMSolverMultistep, TextToImage\n"
                "pipe = TextToImage.from_pretrained('dewml/hybrid-dit-176m')\n"
                "result = pipe(['a lake beneath the northern lights'], key=3, steps=15, "
                "solver=DPMSolverMultistep(), guidance=CFG(6, interval=(0.15, 0.9)))\n"
                "assert result.pil()[0].size == (256, 256)\n"
                "import sys\nassert 'jax' not in sys.modules\n")
        submitted = time.perf_counter()

        def run(kernel, indices):
            identifier, client, _ = kernel
            rows = []
            for index in indices:
                started = time.perf_counter()
                stdout = execute(client,
                    f"assert __measurement == {identifier!r}\n"
                    f"assert Path('/work/context').read_text() == {identifier!r}\n"
                    "from dew.interop import PretrainedDecoder\nfrom dew.sampling import Sampling\n"
                    "model = PretrainedDecoder.load('HuggingFaceTB/SmolLM2-135M-Instruct', "
                    "dtype='float32', max_seq_len=256)\n"
                    "assert model.text_generation(sampling=Sampling(temperature=0))("
                    "'The capital of France is', 24, key=0).text[0]\n"
                    "import json, os\nfrom pathlib import Path\n"
                    "pss = next(line for line in Path('/proc/self/smaps_rollup').read_text().splitlines() "
                    "if line.startswith('Pss:'))\n"
                    "print(json.dumps({'uid': os.getuid(), 'fds': len(os.listdir('/proc/self/fd')), "
                    "'pss_bytes': int(pss.split()[1]) * 1024}))")
                usage = json.loads(stdout)
                assert usage['uid'] >= 6100
                rows.append({"request": index, "execute_seconds": time.perf_counter() - started,
                             "arrival_to_result_seconds": time.perf_counter() - submitted,
                             "uid": usage["uid"], "fds": usage["fds"],
                             "pss_bytes": usage["pss_bytes"]})
            ensure_event_loop().close()
            return rows

        with concurrent.futures.ThreadPoolExecutor(len(kernels)) as pool:
            futures = [pool.submit(run, kernel, range(index, args.count, len(kernels)))
                       for index, kernel in enumerate(kernels)]
            rows = [row for future in futures for row in future.result()]
        assert len({row["uid"] for row in rows}) == len(kernels)
        memory = {row["uid"]: row["pss_bytes"] for row in rows}
        model_processes = []
        for proc in pathlib.Path('/proc').glob('[0-9]*'):
            try:
                status = (proc / 'status').read_text()
                if next(line for line in status.splitlines() if line.startswith('Uid:')).split()[1] != '5000':
                    continue
                command = (proc / 'cmdline').read_bytes()
                if b'model_service.py' in command:
                    model_processes.append(proc.name)
            except (OSError, StopIteration):
                pass
        assert len(model_processes) == 1, model_processes
        print(json.dumps({"requests": args.count, "contexts": len(kernels),
                          "state_isolation": True, "sandbox_checks": True, "all_channels": True,
                          "native_text_and_image": True, "model_processes": model_processes,
                          "startup_seconds": [kernel[2] for kernel in kernels], "kernel_pss_bytes": memory,
                          "kernel_fd_counts": {row["uid"]: row["fds"] for row in rows},
                          "median_execute_seconds": statistics.median(row["execute_seconds"] for row in rows),
                          "arrival_to_result_seconds": [row["arrival_to_result_seconds"] for row in rows]}),
              flush=True)
    finally:
        for identifier, client, _ in kernels:
            client.stop_channels()
            started = time.perf_counter()
            request("/api/kernels/" + identifier, method="DELETE")
            assert time.perf_counter() - started < 5, "kernel shutdown required a forced timeout"
        assert not list(pathlib.Path("/sessions/connections").glob("kernel-*.json"))
        ensure_event_loop().close()


if __name__ == "__main__":
    main()
    training()
