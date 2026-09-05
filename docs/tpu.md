# TPUs

`dew-tpu` creates Cloud TPUs, installs dew on every worker, and starts a recipe
on all of them with one command. It shells out to `gcloud`, so it needs nothing
but an authenticated gcloud and a project with the TPU API enabled.

Every command takes `--dry-run`, which prints the exact commands it would run
and exits. Everything quoted below is real `--dry-run` output for a home
directory of `/home/you`.

## Install

```
pip install "dew-ml @ git+https://github.com/AshishKumar4/dew"
```

## A v5e-16 slice from nothing to a training run

### 1. Defaults, once

```
dew-tpu init --project dew-training --zones us-central2-b,europe-west4-a \
    --accelerator-type v5e-16 --ssh-user you --gcs-bucket dew-data --python-version 3.12
```

```
wrote /home/you/.config/dew/tpu.toml
```

Flags you leave out are asked for, and every field has a flag of the same name.
The file is small enough to edit by hand:

```toml
# dew-tpu defaults. Every field has a flag of the same name.
project = "dew-training"
zones = ["us-central2-b", "europe-west4-a"]
accelerator_type = "v5e-16"
runtime_version = "auto"
ssh_user = "you"
gcs_bucket = "dew-data"
data_disk = ""
python_version = "3.12"
```

`zones` is a search order. When you leave `--zone` off a command, dew-tpu asks
each zone in turn for the TPU and remembers the answer in
`~/.config/dew/zones.json`, so the next command starts with the zone that
worked. A TPU that has moved, or been recreated somewhere else, makes that
zone stop answering and the search runs again from the top.

### 2. Create the slice

```
dew-tpu create dew-16
```

```
gcloud compute tpus tpu-vm create dew-16 --zone=us-central2-b --accelerator-type=v5litepod-16 --version=v2-alpha-tpuv5-lite --project=dew-training
dew-16 in us-central2-b: v5litepod-16 on 2 worker(s)
```

`runtime_version = "auto"` picks the version from the accelerator generation:
`v2-alpha-tpuv6e` for v6e, `v2-alpha-tpuv5` for v5p, `v2-alpha-tpuv5-lite` for
v5e, `tpu-ubuntu2204-base` for v4 and older. `v5e-16` is spelled
`v5litepod-16` on the wire, and dew-tpu translates it.

Without `--dry-run` it polls until the state is READY, then prints the workers:

```
dew-16 in us-central2-b: v5litepod-16 on 2 worker(s)
dew-16 is CREATING
WORKER  INTERNAL IP  EXTERNAL IP
0       10.0.0.1     34.0.0.1
1       10.0.0.2     34.0.0.2
```

Add `--spot` for a preemptible slice, `--queued` to go through the queued
resources API, `--disk NAME` to attach a persistent disk (mounted on the worker
at `/mnt/persist`). A disk is named by its full resource path, so `--disk`
needs a project: `project` in `tpu.toml`, or whatever `gcloud config
get-value project` answers.

### 3. Set up every worker

```
dew-tpu setup dew-16 --from-source --extras tfds
```

```
rsync -az --exclude=.git '--filter=:- .gitignore' -e 'ssh -i /home/you/.ssh/google_compute_engine -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null' /home/you/dew/ 'you@<worker-0-ip>:~/dew/'
rsync -az --exclude=.git '--filter=:- .gitignore' -e 'ssh -i /home/you/.ssh/google_compute_engine -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null' /home/you/dew/ 'you@<worker-1-ip>:~/dew/'
gcloud compute tpus tpu-vm scp /home/you/.config/dew/setup-dew-16.sh 'you@dew-16:~/dew-setup.sh' --zone=us-central2-b --worker=all --project=dew-training
gcloud compute tpus tpu-vm ssh you@dew-16 --zone=us-central2-b --worker=0 '--command=bash ~/dew-setup.sh' --project=dew-training
gcloud compute tpus tpu-vm ssh you@dew-16 --zone=us-central2-b --worker=1 '--command=bash ~/dew-setup.sh' --project=dew-training
gcloud compute tpus tpu-vm ssh you@dew-16 --zone=us-central2-b --worker=0 '--command="$HOME/dew-venv/bin/python" -c '"'"'import jax; print(jax.device_count(), jax.local_device_count())'"'"'' --project=dew-training
gcloud compute tpus tpu-vm ssh you@dew-16 --zone=us-central2-b --worker=1 '--command="$HOME/dew-venv/bin/python" -c '"'"'import jax; print(jax.device_count(), jax.local_device_count())'"'"'' --project=dew-training
```

The setup script is rendered from your flags, copied to every worker and run
there. It installs uv, a Python venv at `~/dew-venv`, `jax[tpu]`, and either the
synced tree in editable mode (`--from-source`) or a release
(`--version 0.2.1`). It raises the open file limit, writes `~/.dew-env` with
`TOKENIZERS_PARALLELISM=false` and `WANDB_CACHE_DIR`, and mounts the bucket from
your config with gcsfuse. Each step checks whether it is already done, so a
second run creates nothing again. The two installs are the exception: they
resolve against PyPI every time, so a `jax[tpu]` or `dew-ml` release since the
last run lands in the venv. Pass `--version` to hold dew-ml at one release.

The last step is the check that matters on a pod slice. Every worker must see
the whole slice.

```
WORKER  DEVICES  LOCAL  CHECK
0       16       8      ok
1       16       8      ok
```

A worker that reports 8 instead of 16 fails the command, which is what you want
before spending an hour on a run that cannot see half its chips.

### 4. Run something everywhere

```
dew-tpu run dew-16 -- python -c 'import jax; print(jax.device_count())'
```

```
gcloud compute tpus tpu-vm ssh you@dew-16 --zone=us-central2-b --worker=0 '--command=. $HOME/.dew-env 2>/dev/null; python -c '"'"'import jax; print(jax.device_count())'"'"'' --project=dew-training
gcloud compute tpus tpu-vm ssh you@dew-16 --zone=us-central2-b --worker=1 '--command=. $HOME/.dew-env 2>/dev/null; python -c '"'"'import jax; print(jax.device_count())'"'"'' --project=dew-training
```

One ssh per worker, in parallel, and every line of output is tagged with the
worker it came from, so `[worker 0]` and `[worker 1]` interleave as they print.
The exit code is the first worker that failed. `--worker 1` talks to one
worker. `--detach` starts the command under `nohup`, writes
`~/dew-runs/<job>/worker-<i>.log`, and prints the job id.

### 5. Train

```
dew-tpu train dew-16 --job lm-1 -- recipes/lm/train.py --data.batch-size 64
```

```
rsync -az --exclude=.git '--filter=:- .gitignore' -e 'ssh -i /home/you/.ssh/google_compute_engine -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null' /home/you/dew/ 'you@<worker-0-ip>:~/dew/'
rsync -az --exclude=.git '--filter=:- .gitignore' -e 'ssh -i /home/you/.ssh/google_compute_engine -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null' /home/you/dew/ 'you@<worker-1-ip>:~/dew/'
gcloud compute tpus tpu-vm ssh you@dew-16 --zone=us-central2-b --worker=0 '--command=mkdir -p $HOME/dew-runs/lm-1 && nohup bash -c '"'"'. $HOME/.dew-env 2>/dev/null; cd $HOME/dew && python recipes/lm/train.py --data.batch-size 64 --trainer.multi-host True'"'"' > $HOME/dew-runs/lm-1/worker-0.log 2>&1 < /dev/null & echo "job lm-1 pid $!"' --project=dew-training
gcloud compute tpus tpu-vm ssh you@dew-16 --zone=us-central2-b --worker=1 '--command=mkdir -p $HOME/dew-runs/lm-1 && nohup bash -c '"'"'. $HOME/.dew-env 2>/dev/null; cd $HOME/dew && python recipes/lm/train.py --data.batch-size 64 --trainer.multi-host True'"'"' > $HOME/dew-runs/lm-1/worker-1.log 2>&1 < /dev/null & echo "job lm-1 pid $!"' --project=dew-training
job lm-1 on 2 worker(s), following worker 0
gcloud compute tpus tpu-vm ssh you@dew-16 --zone=us-central2-b --worker=0 '--command=tail -f -n 200 $HOME/dew-runs/lm-1/worker-0.log' --project=dew-training
```

`train` is sync plus a detached run of the recipe on every worker with
`--trainer.multi-host True`, then it follows worker 0. Ctrl-C stops the
following, not the run. Come back to it with:

```
dew-tpu logs dew-16 lm-1 --follow
```

```
gcloud compute tpus tpu-vm ssh you@dew-16 --zone=us-central2-b --worker=0 '--command=tail -f -n 200 $HOME/dew-runs/lm-1/worker-0.log' --project=dew-training
```

### 6. Look at the slice while it works

```
dew-tpu status dew-16
```

```
dew-16 v5litepod-16 READY HEALTHY in us-central2-b
WORKER  UPTIME  DEW PROCS  DEVICES BUSY  DEVICES
0       3 days  2          1             8
1       3 days  2          1             8
```

`DEVICES BUSY` counts processes holding a `/dev/accel*` device. A crashed run
that left one behind is why `reset` exists: it runs `lsof` on `/dev/accel*` on
every worker and kills what it finds, reporting either the pids it killed or
that nothing held the devices.

```
dew-tpu reset dew-16
```

### 7. Sweeps

```
dew-tpu spawn sweep 3 --type v5e-8 -- python recipes/lm/train.py
```

Three independent v5e-8 TPUs, created in parallel, each set up and each running
the command detached. Output from the three interleaves, tagged by name, and
the summary comes at the end (excerpt):

```
[sweep-0] create
[sweep-1] create
[sweep-2] create
sweep-2 in us-central2-b: v5litepod-8 on 1 worker(s)
[sweep-2] setup
[sweep-1] run sweep-1-20260902-152842
NAME     STATE  JOB
sweep-0  ready  sweep-0-20260902-152842
sweep-1  ready  sweep-1-20260902-152842
sweep-2  ready  sweep-2-20260902-152842
```

No tmux, no dashboard. `dew-tpu list` and `dew-tpu status` answer the same
questions afterwards.

### 8. Clean up

```
dew-tpu delete dew-16
```

```
gcloud compute tpus tpu-vm delete dew-16 --zone=us-central2-b --quiet --project=dew-training
```

A TPU created with `--queued` is held by a queued resource, and delete removes
that instead, which is the only way the node goes away for good.

## The rest of the commands

| Command | What it does |
| --- | --- |
| `dew-tpu init` | Write `~/.config/dew/tpu.toml` from flags, asking for what is missing. |
| `dew-tpu create NAME` | Create a TPU or pod slice, wait for READY, print the workers. |
| `dew-tpu delete NAME` | Delete the TPU, and the queued resource that holds it. |
| `dew-tpu start NAME` / `stop NAME` | Start or stop a TPU without losing it. |
| `dew-tpu list` | Every TPU in every configured zone: name, type, state, health, workers, zone, spot. |
| `dew-tpu describe NAME` | State, health, runtime, topology and the address of each worker. |
| `dew-tpu ssh NAME` | A shell on one worker. `-L 8888` forwards a port, repeat for more. |
| `dew-tpu run NAME -- CMD` | Run CMD on every worker in parallel. `--worker N`, `--detach`. |
| `dew-tpu logs NAME JOB` | The log of a detached job. `--follow`, `--worker all`. |
| `dew-tpu copy NAME SRC DST` | Copy a file or directory to every worker. |
| `dew-tpu sync NAME` | rsync the git working tree to `~/<repo>` on every worker. `--delete` also removes what the local tree does not have. |
| `dew-tpu setup NAME` | uv, venv, `jax[tpu]`, dew, gcsfuse, ulimits, env, then the device check. |
| `dew-tpu train NAME -- RECIPE` | sync, then the recipe on every worker, detached, following worker 0. |
| `dew-tpu status NAME` | Per worker: uptime, dew processes, devices busy. |
| `dew-tpu reset NAME` | Kill whatever holds the accelerators on every worker. |
| `dew-tpu spawn BASE N -- CMD` | N independent TPUs, set up, each running CMD. |

## Coming from tpu_tool.sh

| `tpu_tool.sh` | `dew-tpu` |
| --- | --- |
| `create NAME TYPE [VERSION] [--spot] [--queued] [--no-attach] [--zone Z]` | `create NAME [--type TYPE] [--version V] [--spot] [--queued] [--disk NAME] [--zone Z]`. A disk is attached only when you ask for one. |
| `delete NAME` | `delete NAME`, which also removes a queued resource. |
| `start NAME` / `stop NAME` | `start NAME` / `stop NAME`. These now do something. |
| `list [--zone Z]` | `list [--zone Z]`, with health, worker count and spot in the table. |
| `ssh NAME` | `ssh NAME [--worker N] [-L PORT]`. You name the ports instead of getting ten of them. |
| `update-ssh-config NAME` | Gone. gcloud manages the keys, so nothing writes to your `~/.ssh/config`. |
| `copy-github-key NAME` | Gone. Use `ssh -A` through `ssh NAME -- -A`, or `copy NAME ~/.ssh/key ~/.ssh/key`. |
| `attach-disk NAME DISK` | `create NAME --disk DISK`. Attaching to a live TPU is a `gcloud compute tpus tpu-vm attach-disk` away. |
| `copy NAME SRC DST` | `copy NAME SRC DST [--worker all]`, which reaches every worker. |
| `execute NAME CMD` | `run NAME -- CMD`, which reaches every worker, prefixes the output and returns the first failure. |
| `setup NAME [--mount-gcs B]` | `setup NAME [--from-source] [--version X] [--extras tfds,av] [--gcs-bucket B]`. No miniconda, no pinned jax 0.5.3, and it verifies the device count. |
| `spawn BASE N TYPE CMD` | `spawn BASE N [--type TYPE] -- CMD`, without the tmux session. |
| `reset_tpu.sh` (copied to the TPU by hand) | `reset NAME`, on every worker. |
| `help` | `dew-tpu --help`, and `--help` on any command. |
| `version` | Gone. dew-tpu ships with dew, so `pip show dew-ml` is its version. |

What the old tool did that dew-tpu does not:

- `rm -rf /tmp/*` during reset. It deleted other people's files on a shared
  host and had nothing to do with freeing an accelerator.
- The knot-resolver DNS setup and the fixed nameserver list in `setup_tpu.sh`.
  It belonged to one dataset pipeline on one project.
- torch, tensorflow, diffusers and the rest of the FlaxDiff pin list. `setup`
  installs `jax[tpu]` and dew with the extras you name.
- The tmux dashboard for `spawn`. `list`, `status` and `logs` answer the same
  questions and do not need a terminal multiplexer.
