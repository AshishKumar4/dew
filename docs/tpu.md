# Cloud TPUs

A Cloud TPU slice is a set of TPU chips attached to one or more worker VMs. A job on a multi-worker slice runs the same program on every worker, and the workers join one JAX process pool. `dew tpu` runs `gcloud`, SSH and rsync for you: it creates a slice, installs the environment on each worker and starts a recipe on all of them. To run some other program on every worker as one pool, use `dew launch --tpu NAME` ([Multi-node training](guides/multi-node.md)).

Every resource command on this page has `--dry-run`, so it prints the commands `dew tpu` would run without creating cloud resources or connecting to any worker. A dry run that succeeds has not checked cloud permissions, capacity, networking, TPU execution or distributed training.

## Prerequisites

First do a [local recipe run](recipes.md) and read [distributed training](concepts/distributed.md). Before you remove `--dry-run` from a resource command, have the following ready:

- A Google Cloud project with billing enabled, the Cloud TPU API enabled, and quota for the accelerator type you want in a zone that offers it. Even with quota, capacity can be unavailable.
- An authenticated Google Cloud CLI, and permission to create and delete TPUs, to use the worker service account, and to reach your data or checkpoint buckets. If you get permission errors, check them against the project's IAM policy and grant only what the job needs.
- SSH access to the workers, and `ssh` and `rsync` on your machine. Setup from source and training both run from a Git checkout. The sync step uses the Google Compute Engine SSH key and turns off SSH host-key checking. Decide whether that is acceptable before you use it on a sensitive network.
- Network access from the workers to the package repositories that setup installs from, and passwordless sudo on the workers for setup's package installs and system-limit changes. `dew tpu` does not configure the network, so a private-network deployment needs its own routing and access setup.
- Your dataset files, any model or tokenizer files you need, and a checkpoint location that every process can reach. Sync skips everything that `.gitignore` excludes, so ignored datasets and model weights do not reach the workers that way.

Read [Cloud TPU pricing](https://cloud.google.com/tpu/pricing), and check regional quota and runtime availability for your project. Budget for accelerator time, worker and storage resources, bucket operations, and data transfer. Billing continues after you stop a process or close the log viewer.

Spot TPUs can be preempted. To resume after a preemption you need a checkpoint on storage that every process can reach and a data iterator that saves its position, so read [checkpoints](guides/checkpoints.md) before you choose spot capacity.

Install Dew by following the [installation guide](installation.md). `dew tpu --help` and `dew tpu create --help` show the command's options without contacting Google Cloud. Each command's help ends with the configuration defaults it falls back to and the file they came from.

## Configuration

While you are learning, use a separate configuration directory so you do not overwrite your real deployment defaults:

```bash
export DEW_CONFIG_DIR="$(mktemp -d)"
dew tpu init --project dew-training --zones us-central2-b,europe-west4-a \
    --accelerator-type v5e-16 --runtime-version auto --ssh-user you \
    --gcs-bucket '' --data-disk '' --python-version 3.12 --dry-run
```

`dew-training`, `you`, and `dew-16` are example names. Replace the project and the SSH user with your own before you deploy anything. `init --dry-run` prints the TOML and does not save it. To use the previews below with these example values, save this file as `$DEW_CONFIG_DIR/tpu.toml`:

```toml
project = "dew-training"
zones = ["us-central2-b", "europe-west4-a"]
accelerator_type = "v5e-16"
runtime_version = "auto"
ssh_user = "you"
gcs_bucket = ""
data_disk = ""
python_version = "3.12"
```

For a real configuration, run `dew tpu init` without `--dry-run` and it writes the file for you. In an interactive terminal it asks for any flag you left out. Without `DEW_CONFIG_DIR`, the file is `~/.config/dew/tpu.toml`, or `tpu.toml` under `$XDG_CONFIG_HOME/dew/` when that variable is set.

`zones` is the order in which `dew tpu` searches for an existing TPU. Creation uses `--zone` if you give it, and otherwise the first zone in the list. It does not retry in the next zone when one has no capacity. Dew caches the zone it finds for each TPU in `zones.json`.

Dry runs still write local files. Creation can update the zone cache, and setup writes its generated shell script into the configuration directory, so the separate directory keeps these files out of your normal configuration. No cloud resources are created.

## Creation and setup

```bash
dew tpu create dew-16 --zone us-central2-b --type v5e-16 --dry-run
dew tpu setup dew-16 --zone us-central2-b --type v5e-16 \
    --from-source --dry-run
```

The first command prints:

```text
gcloud compute tpus tpu-vm create dew-16 --zone=us-central2-b --accelerator-type=v5litepod-16 --version=v2-alpha-tpuv5-lite --project=dew-training
dew-16 in us-central2-b: v5litepod-16 on 2 worker(s)
```

Run setup from source inside the Dew Git checkout. Dew turns `v5e-16` into the API name `v5litepod-16`, and the preview assumes two workers.

`runtime_version="auto"` picks a runtime from a table in Dew that has one entry per TPU generation. The table is only a default and does not check what Google offers today, so compare its choice with the current [Cloud TPU software versions](https://cloud.google.com/tpu/docs/runtimes) before you create the slice. `--version` means two different things: on `create` it sets the TPU runtime, and on `setup` it picks the Dew package release.

A real `create` waits until the TPU reports READY. `--spot` asks for capacity that can be preempted, and `--queued` goes through the queued-resources API. `--disk NAME` attaches a persistent disk and mounts it at `/mnt/persist`; `attach-disk NAME DISK` does the same for a TPU that already exists, and `--read-only` lets several workers share one disk. Check the disk's location, permissions, and data ownership before you attach it.

With `--from-source`, setup first syncs the checkout. It then installs uv, creates `~/dew-venv`, installs the checkout with its `tpu` extra (which brings the libtpu for the jax that Dew pins), writes `~/.dew-env`, raises the open-file limits and, if you configured a bucket, mounts it at `~/gcs_mount`. It also installs system packages with sudo.

Without a source flag, setup installs `jax[tpu]` and then the Dew release you chose. Running setup again can pick up newer packages, and pinning Dew does not pin JAX or anything else in the environment, so save the resolved package versions with your run.

`--git-key FILE` copies a private key to every worker as `~/.ssh/id_ed25519` and adds github.com to the trusted hosts, so the workers can clone private repositories over ssh. Use a deploy key that can reach only what the job needs.

A real setup ends by asking each worker for `jax.device_count()` and `jax.local_device_count()`, and it compares the global count with the slice size you asked for. For a two-worker v5e-16, the preview expects 16 global devices and 8 local devices per worker. These are the counts setup expects; they were not read from a deployed slice.

## Checking a deployed slice

After a real setup, read the report from every worker. READY means the resource exists. It does not mean your training program can finish a collective operation. The device-count check does not test checkpoint access, input throughput, or your model's sharding layout either.

To preview a command on all workers:

```bash
dew launch --tpu dew-16 --zone us-central2-b --dry-run -- \
    python -c 'import jax; print(jax.process_index(), jax.process_count(), jax.device_count(), jax.local_device_count())'
```

`dew launch --tpu` starts one SSH command per worker. Each command sources `~/.dew-env` first, each line of output is prefixed with its worker's rank, and when one worker fails the launcher stops the rest. [Multi-node training](guides/multi-node.md) covers `--cwd` and multislice runs. Your program still has to initialize distributed JAX before it creates device arrays; the training recipes do that. The `python -c` command above only shows what JAX sees in each process, so it does not test distributed training.

Before you trust a deployment, run a short recipe with the mesh and global batch size you plan to use. Check that every process joins, that updates are finite, and that every process can read the data and write to the checkpoint location. Then restore from that checkpoint in a second short run. None of the previews on this page do any of that on real hardware.

## Data and training jobs

The [recipe walkthrough](recipes.md) creates `/tmp/dew-first-recipe/tokens`. Copy that directory to the workers yourself:

```bash
dew tpu copy dew-16 /tmp/dew-first-recipe/tokens '~/dew-tokens' \
    --zone us-central2-b --dry-run
```

`copy` sends to every worker unless you pass `--worker`. For the example SSH user the data goes to `/home/you/dew-tokens`; change that path if your home directory on the workers is different. Go back to the Dew checkout and preview a short launch:

```bash
dew tpu train dew-16 --zone us-central2-b --job byte-demo --dry-run -- \
    recipes/lm/train.py data:token-windows \
    --data.path /home/you/dew-tokens --data.seq-len 16 \
    --data.loading.workers 0 --data.loading.threads 1 \
    --data.loading.read-buffer 2 --data.val-batches 2 \
    --model.emb-features 16 --model.num-layers 1 --model.num-heads 2 --model.mlp-features 32 \
    --trainer.batch-size 32 --trainer.steps 2 --trainer.log-every 1 \
    --trainer.eval-every 2 --trainer.checkpoint-every 2 \
    --trainer.checkpoint-dir /home/you/dew-checkpoints \
    --trainer.name byte-demo --max-new-tokens 0
```

The model is tiny so that you can inspect the launch; its settings say nothing about TPU performance. `train` syncs the checkout, adds `--trainer.multi-host True` and starts the recipe in the background on every worker, then follows worker 0's log.

In this example each worker writes checkpoints to its own local disk. The same local path on two workers is usually two different disks, so this is not durable shared storage for a multi-host run. Before a real run, pick a checkpoint store that every process can reach, set up its credentials and test it.

Ctrl-C stops following the log, and the training job keeps running. The logs are in `~/dew-runs/<job>/worker-<index>.log` on each worker. To preview the commands for checking on it later:

```bash
dew tpu logs dew-16 byte-demo --zone us-central2-b --worker all --dry-run
dew tpu status dew-16 --zone us-central2-b --dry-run
dew tpu describe dew-16 --zone us-central2-b --dry-run
```

`logs` reads worker 0 unless you pass `--worker N` or `--worker all`, and `--follow` keeps printing new lines. When something fails, read the logs from every worker. A launch command that succeeds means only that the job started.

## Deleting and resetting

List or describe your resources before you delete them, and save any outputs you need. To preview deletion:

```bash
dew tpu delete dew-16 --zone us-central2-b --dry-run
```

A real `delete` checks whether a queued resource owns the TPU, and if so deletes the queued resource. A dry run cannot look up that ownership, so the deletion command it prints can differ from the real one. Afterward, check your project to confirm the deletion, including disks and buckets, which have their own lifetimes and charges.

`reset` kills every process that holds the accelerator devices on every worker. Use it to recover a stuck slice, not to stop a job normally. On a shared slice it can kill someone else's work, and anything unsaved is lost. `stop` keeps the TPU's name and disks. After either command, check what you are still being billed for.

## Command reference

| Command | Purpose |
| --- | --- |
| `init` | Write `tpu.toml` from flags, asking for what is missing in an interactive terminal. |
| `create NAME` | Create a TPU VM or pod slice and wait until it is READY. `--spot`, `--queued`, `--disk`. |
| `setup NAME` | Install uv, a virtualenv, JAX and Dew on every worker, then count the devices. `--from-source`, `--git-key`, `--gcs-bucket`. |
| `sync NAME` | Copy the git working tree to `~/<repo>` on every worker; `--delete` also removes remote files absent locally. |
| `copy NAME SRC DEST` | Copy a local file or directory to every worker, or one with `--worker N`. |
| `train NAME -- RECIPE ...` | Sync the tree and start a recipe on every worker in the background, then follow worker 0's log. |
| `logs NAME JOB` | Show a detached job's log; `--worker N` or `all`, `--follow`. |
| `status NAME` | Show what each worker is doing. |
| `run NAME -- CMD` | Run a shell command on every worker, or one with `--worker N`; `--detach` runs it in the background. |
| `list` | List TPUs in the configured zones, with worker 0's external IP. |
| `describe NAME` | Show what a TPU is and where its workers are. |
| `ssh NAME --worker N -L PORT` | Open a worker shell with an optional local port forward. |
| `ssh-config NAME` | Write `~/.ssh/config` entries: `NAME` for worker 0, `NAME-worker-N` for the others. `delete` removes them. |
| `attach-disk NAME DISK` | Attach a persistent disk to an existing TPU and mount it at `/mnt/persist` on every worker. |
| `start NAME` / `stop NAME` | Change a TPU's running state without deleting its name and disks. |
| `reset NAME` | Kill whatever holds the accelerators on every worker. |
| `delete NAME` | Delete a TPU, and the queued resource that holds it. |
| `spawn BASE N -- CMD` | Create, set up and launch on N independent TPUs. Each adds cost. |

Read each command's `--help` before passing an option that deletes or kills something.

### tpu_tool.sh equivalents

If you used the old `tpu_tool.sh`, its `create`, `delete`, `start`, `stop`, `list`, `ssh`, `copy` and `setup` commands keep their names. `execute` is now `run`, `update-ssh-config` is `ssh-config`, `copy-github-key` is `setup --git-key` and `--mount-gcs` is `setup --gcs-bucket`, and `reset` does what the reset script did. As before, every command looks for the TPU across the configured zones. When the workers need the credentials in `~/.netrc`, send the file with `copy`. `spawn` prints each TPU's progress next to its name and ends with a table, where the old tool opened a tmux dashboard; run it inside tmux if you want to detach. Dew does not clear `/tmp`, does not install the old framework's conda environment or its pinned package list, and does not rewrite the project's DNS configuration.
