"""Administrative Kernel Gateway startup/memory measurements on shared compute."""

import argparse
import concurrent.futures
import json
import pathlib
import statistics
import time
import urllib.request

from jupyter_client import BlockingKernelClient


def dump_processes():
    for proc in pathlib.Path('/proc').glob('[0-9]*'):
        try:
            command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            text = (proc / 'status').read_text() + '\ncmd=' + command
            if '6100' in text or '6101' in text:
                print('PROCESS_STATUS',text[:1600],flush=True)
        except OSError:
            pass

def execute(client, code):
    message = client.execute(code, allow_stdin=False)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("count", type=int, choices=(1, 10, 50))
    args = parser.parse_args()
    token = pathlib.Path("/run/dew/gateway-token").read_text()

    def request(path, data=None, method=None):
        body = None if data is None else json.dumps(data).encode()
        req = urllib.request.Request("http://127.0.0.1:8890" + path, data=body, method=method,
                                     headers={"Authorization": "token " + token,
                                              "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=40) as response:
            raw = response.read()
            return json.loads(raw) if raw else None

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
        submitted = time.perf_counter()

        def run(kernel, indices):
            identifier, client, _ = kernel
            rows = []
            for index in indices:
                started = time.perf_counter()
                stdout = execute(client,
                    f"assert __measurement == {identifier!r}\n"
                    f"assert Path('/work/context').read_text() == {identifier!r}\n"
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
            return rows

        with concurrent.futures.ThreadPoolExecutor(len(kernels)) as pool:
            futures = [pool.submit(run, kernel, range(index, args.count, len(kernels)))
                       for index, kernel in enumerate(kernels)]
            rows = [row for future in futures for row in future.result()]
        assert len({row["uid"] for row in rows}) == len(kernels)
        memory = {row["uid"]: row["pss_bytes"] for row in rows}
        print(json.dumps({"requests": args.count, "contexts": len(kernels), "state_isolation": True,
                          "startup_seconds": [kernel[2] for kernel in kernels], "kernel_pss_bytes": memory,
                          "kernel_fd_counts": {row["uid"]: row["fds"] for row in rows},
                          "median_execute_seconds": statistics.median(row["execute_seconds"] for row in rows),
                          "arrival_to_result_seconds": [row["arrival_to_result_seconds"] for row in rows]}),
              flush=True)
    finally:
        for identifier, client, _ in kernels:
            client.stop_channels()
            request("/api/kernels/" + identifier, method="DELETE")
        assert not list(pathlib.Path("/sessions/connections").glob("kernel-*.json"))


if __name__ == "__main__":
    main()
