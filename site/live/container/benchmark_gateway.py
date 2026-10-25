"""Administrative Kernel Gateway startup/memory measurements on shared compute."""

import argparse
import concurrent.futures
import json
import pathlib
import queue
import statistics
import sys
import time
import urllib.request

from guests import GUEST_UIDS
from jupyter_client import BlockingKernelClient
from jupyter_core.utils import ensure_event_loop
from shared_bridge import OOM_PROBE, load, processes


def dump_processes():
    for _, fields, command in processes():
        text = "\n".join(f"{name}:{value}" for name, value in fields.items()) + "\ncmd=" + (
            command.replace(b"\0", b" ").decode(errors="replace"))
        if any(str(uid) in text for uid in GUEST_UIDS[:2]):
            print('PROCESS_STATUS', text[:1600], flush=True)

def request(path, data=None, method=None):
    token = pathlib.Path("/run/dew/gateway-token").read_text()
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request("http://127.0.0.1:8890" + path, data=body, method=method,
                                 headers={"Authorization": "token " + token,
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=40) as response:
        raw = response.read()
        return json.loads(raw) if raw else None


def ready(client):
    """Wait for `client`'s kernel to start; one that dies first fails with the gateway's log,
    where its launcher says why."""
    try:
        client.wait_for_ready(timeout=30)
    except RuntimeError as error:
        log = pathlib.Path("/run/dew/gateway.log").read_text(errors="replace").splitlines()[-30:]
        raise RuntimeError(f"{error}; gateway log:\n" + "\n".join(log)) from None


def execute(client, code):
    return collect(client, client.execute(code, allow_stdin=False))


class Silent(TimeoutError):
    """The kernel sent nothing for the timeout; `output` is what it had sent."""

    def __init__(self, output):
        super().__init__(f"the kernel went silent after: {output[-1500:]!r}")
        self.output = output


def collect(client, message, timeout=30):
    stdout = ""
    while True:
        try:
            result = client.get_iopub_msg(timeout=timeout)
        except queue.Empty:
            raise Silent(stdout) from None
        if result["parent_header"].get("msg_id") != message:
            continue
        if result["msg_type"] == "stream":
            stdout += result["content"]["text"]
        if result["msg_type"] == "error":
            raise RuntimeError(str(result["content"]))
        if result["msg_type"] == "status" and result["content"]["execution_state"] == "idle":
            return stdout


# A cell's caps hold from the cell after the one that installs them (live_training.install).
CAPS = "import live_training\nlive_training.install()"
TRAINING_PROBE = OOM_PROBE + pathlib.Path(__file__).with_name("training-probe.py").read_text()


def training():
    """A training context reaches neither the network nor any path outside its scratch and its
    private overlay of the prepared cache, reads that cache offline, and says first what its
    caps changed."""
    kernel = request("/api/kernels", {"name": "dew-train"})
    client = BlockingKernelClient(connection_file=f"/run/dew/gateway/kernel-{kernel['id']}.json")
    client.load_connection_file()
    client.start_channels()
    try:
        ready(client)
        execute(client, CAPS)
        lines = execute(client, TRAINING_PROBE).splitlines()
        assert lines[0] == "Live run: batch 64 -> 8, so it fits a shared 4-vCPU host.", lines
        assert lines[1] == ("Live run: steps 1000 -> 20, log_every 100 -> 5, "
                            "so it finishes in about two minutes."), lines
    finally:
        client.stop_channels()
        request("/api/kernels/" + kernel["id"], method="DELETE")
    assert not pathlib.Path("/opt/train/escape").exists(), "a training context wrote the prepared cache"


def cells():
    """The page's pool cells and its train.py, each run in a fresh training context as a
    visitor's Run does, while the model process holds its models. The three that hold a few
    GiB run one after another beside the small ones, to end within the smoke's 15-minute
    alarm (site/live/src/preparer.ts). A failure names the host's memory."""
    sys.path.insert(0, "/opt/live/cells")
    import cells as page
    programs = {name: page.cell(name) for name in page.CELLS["pool"]}
    programs["hero"] = page.training_example(pathlib.Path("/opt/live/cells/hero.py").read_text())
    large = [name for name in ("decide", "finetune", "hero") if name in programs]
    streams = [large, [name for name in programs if name not in large]]
    with concurrent.futures.ThreadPoolExecutor(len(streams)) as pool:
        for future in [pool.submit(lambda names: [cell(name, programs[name]) for name in names], names)
                       for names in streams]:
            future.result()


def cell(name, code):
    # The bridge installs the caps in every training context first, as a cell of its own (PRELOAD,
    # shared_bridge.py); the last line, printed only when the cell finished, is what the run kept
    # resident.
    code = (code + "\nimport resource\n"
            "print('peak', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss >> 10)\n")
    kernel = request("/api/kernels", {"name": "dew-train"})
    client = BlockingKernelClient(connection_file=f"/run/dew/gateway/kernel-{kernel['id']}.json")
    client.load_connection_file()
    client.start_channels()
    started = time.perf_counter()
    try:
        ready(client)
        execute(client, CAPS)
        output = collect(client, client.execute(code, allow_stdin=False), timeout=240)
    except Exception as error:
        raise AssertionError(f"{name} failed after {time.perf_counter() - started:.0f} s: {error}; "
                             f"memory {memory()}; load {load()}") from None
    finally:
        client.stop_channels()
        request("/api/kernels/" + kernel["id"], method="DELETE")
    # What a run keeps resident, which the host's memory must hold beside the model process.
    peak = int(output.rsplit("peak ", 1)[1].split()[0])
    assert peak < 4608, f"{name} held {peak} MiB; memory {memory()}"
    seconds = time.perf_counter() - started
    print(f"{name} ran in {seconds:.0f} s, {peak} MiB resident; memory {memory()}", flush=True)


def memory():
    lines = dict(line.split(":", 1) for line in pathlib.Path("/proc/meminfo").read_text().splitlines())
    return {key: int(lines[key].split()[0]) >> 10 for key in ("MemTotal", "MemAvailable", "Shmem")}


def start_contexts(kernels, count):
    """Start up to eight contexts into `kernels`, each marked as its own, timing its start."""
    for _ in range(min(count, 8)):
        start = time.perf_counter()
        model = request("/api/kernels", {"name": "python3"})
        connection = pathlib.Path("/run/dew/gateway") / f"kernel-{model['id']}.json"
        client = BlockingKernelClient(connection_file=str(connection))
        client.load_connection_file()
        client.start_channels()
        kernels.append((model["id"], client, 0))
        try:
            ready(client)
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


def isolated(kernels):
    """Every channel of each context answers its own requests, and a guest reaches nothing outside it."""
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


def warmed(kernels):
    """Each context samples an image through the shared model, without importing JAX."""
    for _, client, _ in kernels:
        execute(client,
            "import model_client\nmodel_client.install()\n"
            "from dew.inference import TextToImage\n"
            "from dew.sampling import CFG, DPMSolverMultistep\n"
            "pipe = TextToImage.from_pretrained('dewml/hybrid-dit-176m')\n"
            "result = pipe(['a lake beneath the northern lights'], key=3, steps=15, "
            "solver=DPMSolverMultistep(), guidance=CFG(6, interval=(0.15, 0.9)))\n"
            "assert result.pil()[0].size == (256, 256)\n"
            "import sys\nassert 'jax' not in sys.modules\n")


def measured(kernels, count):
    """`count` text requests spread over the contexts, each context's from a thread of its own."""
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
            assert usage['uid'] in GUEST_UIDS
            rows.append({"request": index, "execute_seconds": time.perf_counter() - started,
                         "arrival_to_result_seconds": time.perf_counter() - submitted,
                         "uid": usage["uid"], "fds": usage["fds"],
                         "pss_bytes": usage["pss_bytes"]})
        ensure_event_loop().close()
        return rows

    with concurrent.futures.ThreadPoolExecutor(len(kernels)) as pool:
        futures = [pool.submit(run, kernel, range(index, count, len(kernels)))
                   for index, kernel in enumerate(kernels)]
        return [row for future in futures for row in future.result()]


def report(kernels, rows, count):
    """Every context held its own uid and one model process served them, as one line of JSON."""
    assert len({row["uid"] for row in rows}) == len(kernels)
    memory = {row["uid"]: row["pss_bytes"] for row in rows}
    model_processes = [str(pid) for pid, fields, command in processes()
                       if fields["Uid"].split()[1] == "5000" and b"model_service.py" in command]
    assert len(model_processes) == 1, model_processes
    print(json.dumps({"requests": count, "contexts": len(kernels),
                      "state_isolation": True, "sandbox_checks": True, "all_channels": True,
                      "native_text_and_image": True, "model_processes": model_processes,
                      "startup_seconds": [kernel[2] for kernel in kernels], "kernel_pss_bytes": memory,
                      "kernel_fd_counts": {row["uid"]: row["fds"] for row in rows},
                      "median_execute_seconds": statistics.median(row["execute_seconds"] for row in rows),
                      "arrival_to_result_seconds": [row["arrival_to_result_seconds"] for row in rows]}),
          flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("count", type=int, choices=(1, 10, 50))
    args = parser.parse_args()
    kernels = []
    try:
        start_contexts(kernels, args.count)
        isolated(kernels)
        warmed(kernels)
        report(kernels, measured(kernels, args.count), args.count)
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
    cells()
